from flask import Flask, request, render_template  # Impor Flask dan util templating
import cloudpickle  # Untuk memuat model/pipeline yang dipickle
import os, traceback  # Utilitas OS dan pelacakan error
import re  # Ekspresi reguler
import numpy as np  # Operasi numerik (probabilitas, softmax, dsb.)

# === Preprocessing functions ===
def remove_html_and_content(code: str) -> str:  # Hapus tag HTML umum beserta isinya
    html_tags = [  # Daftar tag HTML yang akan disaring
        "a","abbr","address","area","article","aside","audio","b","base",
        "bdi","bdo","blockquote","body","br","button","canvas","caption",
        "cite","code","col","colgroup","data","datalist","dd","del",
        "details","dfn","dialog","div","dl","dt","em","embed","fieldset",
        "figcaption","figure","footer","form","h1","h2","h3","h4","h5",
        "h6","head","header","hr","html","i","iframe","img","input",
        "ins","kbd","label","legend","li","link","main","map","mark",
        "meta","meter","nav","noscript","object","ol","optgroup","option",
        "output","p","param","picture","pre","progress","q","rp","rt",
        "ruby","s","samp","script","section","select","small","source",
        "span","strong","style","sub","summary","sup","table","tbody",
        "td","template","textarea","tfoot","th","thead","time","title",
        "tr","track","u","ul","var","video","wbr"
    ]
    tags_pattern = '|'.join(html_tags)  # Gabungkan jadi pola alternasi untuk regex

    content_block_pattern = re.compile(  # Pola untuk <tag>...</tag> (berpasangan)
        rf'<(?:{tags_pattern})\b[^>]*?>.*?</(?:{tags_pattern})>',
        flags=re.IGNORECASE | re.DOTALL  # Abaikan kapitalisasi dan tangkap multiline
    )
    for match in content_block_pattern.findall(code):  # Iterasi semua blok yang cocok
        if ".$" not in match:  # Pengecualian kecil (sesuai logika awal)
            code = code.replace(match.strip(), '')  # Hapus blok dari teks

    self_closing_pattern = re.compile(  # Pola untuk tag tunggal/mandiri
        rf'<(?:{tags_pattern})\b[^>]*/?>',
        flags=re.IGNORECASE
    )
    for match in self_closing_pattern.findall(code):  # Iterasi tag mandiri
        if ".$" not in match:  # Pengecualian kecil
            code = code.replace(match.strip(), '')  # Hapus tag dari teks

    return code.strip()  # Kembalikan kode yang sudah dibersihkan

def process_php_tokenized(code: str) -> str:  # Ambil dan rapikan blok PHP dari input
    php_blocks = re.findall(r'(<\?(?:php|=)(?:.|\n)*?\?>)', code)  # Tangkap semua blok <?php ... ?>
    php_code = '\n'.join(php_blocks)  # Satukan menjadi satu string

    open_count = len(re.findall(r'<\?(php|=)', php_code, re.IGNORECASE))  # Hitung pembuka PHP
    close_count = len(re.findall(r'\?>', php_code))  # Hitung penutup PHP
    if open_count > close_count:  # Jika pembuka lebih banyak
        php_code += '\n' + '?>' * (open_count - close_count)  # Tambahkan penutup untuk menyeimbangkan

    php_code = re.sub(r'/\*[\s\S]*?\*/', '', php_code)  # Hapus komentar blok /* ... */
    php_code = re.sub(r'//.*', '', php_code)  # Hapus komentar // baris
    php_code = re.sub(r'#.*', '', php_code)  # Hapus komentar # baris

    php_code = remove_html_and_content(php_code)  # Hapus elemen HTML tersisa dalam blok PHP
    return php_code.strip()  # Kembalikan kode PHP bersih

# === Canonicalisasi variabel ke: $tainted, $sanitized, $dataflow, $context ===
_CANONICAL = ["$tainted", "$sanitized", "$dataflow", "$context"]

# Superglobal dan variabel khusus yang harus dipertahankan
_RESERVED_VARS = {
    '$GLOBALS', '$_GET', '$_POST', '$_REQUEST', '$_COOKIE', '$_SERVER',
    '$_FILES', '$_ENV', '$_SESSION', '$this'
}
_RESERVED_VARS_LC = {v.lower() for v in _RESERVED_VARS}

# $var pattern (hindari variable variables $$x dengan negative lookbehind)
_VAR_RE = re.compile(r'(?<!\$)\$[A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*')
# String literal masker (single/double, dengan escape)
_STR_RE = re.compile(r"""('(?:\\'|[^'])*'|"(?:\\"|[^"])*")""", re.DOTALL)

def _mask_strings(text):
    placeholders = {}
    def repl(m):
        key = f"__STR_{len(placeholders)}__"
        placeholders[key] = m.group(0)
        return key
    return _STR_RE.sub(repl, text), placeholders

def _unmask_strings(text, placeholders):
    # Ganti dari kunci terpanjang untuk mencegah overlap
    for k in sorted(placeholders.keys(), key=len, reverse=True):
        text = text.replace(k, placeholders[k])
    return text

def _heuristic_bucket(name_lc):
    # Heuristik sederhana agar nama asli memandu pemetaan kanonik
    if any(w in name_lc for w in [
        'input','in','param','arg','raw','user','usr','req','get','post','cookie','querystring','stdin','read'
    ]):
        return "$tainted"
    if any(w in name_lc for w in [
        'clean','safe','sanitize','sanit','filter','escape','esc','valid','strip','encode','purify'
    ]):
        return "$sanitized"
    if any(w in name_lc for w in [
        'ctx','context','sql','stmt','statement','query','where','select','insert','update','delete','db','conn'
    ]):
        return "$context"
    if any(w in name_lc for w in [
        'data','flow','tmp','buf','mid','res','value','val','content','payload'
    ]):
        return "$dataflow"
    return None

def normalize_php_variables_to_canonical(code: str) -> str:
    # 1) Masker string agar isi string tidak ikut diubah
    masked, placeholders = _mask_strings(code)

    # 2) Kumpulkan variabel dalam urutan kemunculan pertama
    seen = []
    for m in _VAR_RE.finditer(masked):
        v = m.group(0)
        if v in _RESERVED_VARS or v.lower() in _RESERVED_VARS_LC:
            continue
        if v not in seen:
            seen.append(v)

    # 3) Tentukan pemetaan -> token kanonik
    mapping = {}
    used_canon = set()

    # 3a) Heuristik berdasarkan nama asli
    for v in seen:
        name_lc = v[1:].lower()
        bucket = _heuristic_bucket(name_lc)
        if bucket and bucket not in used_canon:
            mapping[v] = bucket
            used_canon.add(bucket)

    # 3b) Isi slot kanonik yang belum terpakai sesuai urutan tetap
    for canon in _CANONICAL:
        if canon in used_canon:
            continue
        for v in seen:
            if v not in mapping:
                mapping[v] = canon
                used_canon.add(canon)
                break

    # 3c) Variabel sisanya -> $varN (agar stabil tapi tidak “mengganggu” fitur)
    extra_idx = 1
    for v in seen:
        if v not in mapping:
            mapping[v] = f"$var{extra_idx}"
            extra_idx += 1

    # 4) Terapkan pemetaan
    if mapping:
        def repl(m):
            v = m.group(0)
            return mapping.get(v, v)
        masked = _VAR_RE.sub(repl, masked)

    # 5) Kembalikan string yang sudah diunmask
    return _unmask_strings(masked, placeholders)

# === Helper untuk probabilitas semua kelas ===
def _get_classes(model):  # Ambil daftar kelas dari model/pipeline
    # Coba ambil dari pipeline langsung
    cls = getattr(model, "classes_", None)  # Atribut umum pada estimator scikit-learn
    if cls is not None:
        return cls  # Jika ada, langsung kembalikan
    # Coba dari estimator terakhir di Pipeline
    try:
        return model.steps[-1][1].classes_  # Ambil dari step estimator terakhir
    except Exception:
        return None  # Jika gagal, None

def predict_with_probs(pipeline, text: str):  # Prediksi label dan probabilitas
    """Kembalikan (pred_label, classes, probs_np) untuk satu input teks."""
    X = [text]  # Bungkus input jadi list untuk scikit-learn
    # Prediksi label utama
    try:
        pred_label = pipeline.predict(X)[0]  # Prediksi kelas
    except Exception:
        pred_label = None  # Jika gagal, tandai None

    classes = _get_classes(pipeline)  # Ambil daftar kelas dari pipeline

    # 1) Gunakan predict_proba jika ada
    try:
        proba = pipeline.predict_proba(X)[0]  # Probabilitas per kelas
        if classes is None:
            classes = _get_classes(pipeline)  # Pastikan kelas terisi
    except Exception:
        # 2) Gunakan decision_function lalu sigmoid/softmax
        try:
            scores = pipeline.decision_function(X)  # Skor margin (SVM/logit)
            scores = np.atleast_2d(scores)[0]  # Pastikan bentuk array 1D
            if np.ndim(scores) == 0 or (isinstance(scores, np.ndarray) and scores.shape == ()):
                # Binary case satu skor
                s = float(np.ravel(scores)[0])  # Ambil skalar
                if classes is None:
                    classes = np.array([0, 1])  # Default 2 kelas
                p_pos = 1.0 / (1.0 + np.exp(-s))  # Sigmoid untuk kelas positif
                proba = np.array([1.0 - p_pos, p_pos])  # Prob negatif dan positif
            else:
                # Multikelas
                if classes is None:
                    classes = _get_classes(pipeline) or np.arange(len(scores))  # Default index kelas
                exp_s = np.exp(scores - np.max(scores))  # Stabilkan sebelum softmax
                proba = exp_s / exp_s.sum()  # Softmax
        except Exception:
            # 3) Fallback terakhir: one-hot di pred_label
            if classes is None:
                classes = np.array([pred_label]) if pred_label is not None else np.array([0, 1])  # Default kelas
            if pred_label is None:
                # Tidak ada prediksi, rata
                proba = np.ones(len(classes)) / len(classes)  # Distribusi seragam
            else:
                proba = np.array([1.0 if c == pred_label else 0.0 for c in classes])  # One-hot

    if classes is None:
        classes = np.arange(len(proba))  # Jika masih None, gunakan indeks

    # Pastikan tipe numpy array float
    proba = np.asarray(proba, dtype=float)  # Konversi ke float
    return pred_label, classes, proba  # Kembalikan hasil lengkap

# === Flask App ===
app = Flask(__name__)  # Inisialisasi aplikasi Flask

# MODEL_FILENAME = 'dtc_pipeline_FINTUN_cv.pkl'
MODEL_FILENAME = "lsvc_pipeline_FINTUN_cv.pkl"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # Direktori file saat ini
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'tuned_cv', MODEL_FILENAME)  # Lokasi model

try:
    with open(MODEL_PATH, 'rb') as f:  # Buka file model dalam mode baca biner
        pipeline = cloudpickle.load(f)  # Muat pipeline yang terserialisasi
except FileNotFoundError:
    raise RuntimeError(f"Could not find model at {MODEL_PATH}")  # Beri tahu jika model tidak ditemukan

@app.errorhandler(Exception)
def all_errors(e):  # Handler global untuk error tak tertangani
    tb = traceback.format_exc()  # Ambil stack trace sebagai string
    return render_template(  # Render halaman dengan info error
        "index.html",
        code=request.form.get('code',''),  # Isi kembali input pengguna jika ada
        prediction=None,  # Tidak ada prediksi saat error
        probs=[],  # Kosongkan probabilitas
        error=tb  # Tampilkan stack trace (untuk debug)
    ), 500  # Kode status HTTP 500

@app.route('/', methods=['GET','POST'])
def index():  # Route utama untuk form dan hasil
    code_input = ''  # Inisialisasi input kode
    prediction = None  # Inisialisasi label prediksi
    error = None  # Inisialisasi pesan error
    probs = []  # list of dict: [{'label': ..., 'percent': ..., 'is_pred': ...}, ...]  # Struktur data probabilitas

    if request.method == 'POST':  # Jika form disubmit
        code_input = request.form['code']  # Ambil teks kode dari form
        try:
            cleaned_code = process_php_tokenized(code_input)  # Preproses dan ekstrak blok PHP

            # === Tambahan: samakan nama variabel ke token kanonik dataset ===
            cleaned_code = normalize_php_variables_to_canonical(cleaned_code)

            # Prediksi dan probabilitas tiap kelas
            pred_raw, classes, proba = predict_with_probs(pipeline, cleaned_code)  # Lakukan inferensi

            # Susun daftar probabilitas untuk ditampilkan (urut desc)
            pairs = list(zip(classes, proba))  # Gabungkan kelas dan proba
            pairs.sort(key=lambda x: float(x[1]), reverse=True)  # Urutkan menurun
            probs = [
                {
                    'label': str(lbl),  # Nama/ID kelas
                    'percent': float(p) * 100.0,  # Konversi ke persen
                    'is_pred': (str(lbl) == str(pred_raw))  # Penanda kelas prediksi utama
                }
                for lbl, p in pairs
            ]

            # Mapping label
            raw_lower = str(pred_raw).lower() if pred_raw is not None else ''  # Normalisasi string label
            if ('select' in raw_lower) or ('sql' in raw_lower):  # Heuristik untuk SQLi
                prediction = 'SQL Injection Vulnerable'  # Label akhir untuk SQLi
            elif ('script' in raw_lower) or ('xss' in raw_lower):  # Heuristik untuk XSS
                prediction = 'XSS Vulnerable'  # Label akhir untuk XSS
            else:
                prediction = 'Invulnerable to XSS and SQL Injection'  # Jika tidak cocok, dianggap aman

        except Exception:
            error = traceback.format_exc()  # Simpan stack trace jika gagal

    return render_template(  # Render halaman hasil atau form awal
        "index.html",
        code=code_input,  # Kembalikan input pengguna ke form
        prediction=prediction,  # Label prediksi akhir
        probs=probs,  # Daftar probabilitas kelas
        error=error  # Pesan error jika ada
    )

if __name__ == '__main__':  # Entry point aplikasi
    app.run(host='0.0.0.0', port=5000, debug=True)  # Jalankan server Flask (debug aktif)
