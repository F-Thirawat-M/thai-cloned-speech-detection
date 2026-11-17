# งานวิจัย RM: การตรวจจับเสียงโคลนภาษาไทย (SCS4-15)

แผนฉบับเต็ม: https://claude.ai/artifact/L1UgGhUmsBr1bLSHAdsC7h

โฟลเดอร์นี้เป็นงานรอบใหม่ของวิชา RM โปรเจกต์เก่าที่ `year3_semester1/CP353761/project` ยังอยู่ครบ ไม่ได้แก้หรือลบ

ข้อมูลที่ copy มาจากโปรเจกต์เก่า (ไม่ commit):
- `data/raw/common_voice/cv-corpus-27.0-2026-09-11/th/` Common Voice Scripted Speech 27.0 ภาษาไทยทั้งชุด (366,656 คลิป)
- `data/old_pilot/real_all.csv` รายการ pilot เดิม ใช้ตัดผู้พูดและประโยคที่เคยใช้ออก

copy ทั้งชุดด้วย robocopy (รันซ้ำได้ ไฟล์ที่มีแล้วจะข้าม):

```powershell
robocopy "D:\KKU_files\year3_semester1\CP353761\project\data\raw\common_voice" "D:\KKU_files\year3_semester2\CP353764\thai-cloned-speech-detection\data\raw\common_voice" /E /MT:16 /XO /NFL /NDL
```

## ลำดับงาน

| ขั้น | สคริปต์ | สถานะ |
| --- | --- | --- |
| 1 คัดผู้พูดและแบ่งชุด | `scripts/01_select_cohort_split.py` | เสร็จ ใช้ `data/cohort/cohort_v3` (v1, v2 เก็บไว้เฉย ๆ) |
| 2 ทดลองสร้างเสียงนำร่อง 4 ระบบ | `02_select_pilot_items.py`, `03_pilot_omnivoice.py`, `04_pilot_f5_thai.py` | OmniVoice ผ่าน (~2 วิ/คลิป, VRAM ~2 GB); F5-TTS-THAI ผ่าน (~2 วิ/คลิป, VRAM ~0.8 GB); ใช้แบบ B ทั้งคู่; ThonburianTTS ตัดออก (ดึงคำจากเสียงต้นแบบมาเติมบ่อย); OpenJaiTTS, Seed-VC ยังไม่ทำ |
| 3 สร้างเสียงโคลนชุดหลัก + ตรวจ CER/SIM | ยังไม่ทำ | |
| 4 เตรียมเสียงร่วม + ตรวจทางลัด | ยังไม่ทำ | |
| 5 ทดสอบการฟัง (กลุ่ม listen) | ยังไม่ทำ | |
| 6 Fine-tune E1 และ E2 (leave-one-family-out) | ยังไม่ทำ | |
| 7 ประเมิน Test ครั้งเดียว + E4 | ยังไม่ทำ | |

## ขั้น 1: คัดผู้พูดและแบ่งชุด

```powershell
python scripts\01_select_cohort_split.py `
  --cv-dir data\raw\common_voice\cv-corpus-27.0-2026-09-11\th `
  --exclude-manifest data\old_pilot\real_all.csv `
  --run-id cohort_v4
```

ต้องมี pandas และ numpy ค่าเริ่มต้น: seed 2026, เสียงจริง 3,000 คลิป (1 คลิปต่อผู้พูด),
Train/Validation/Test = 70/20/10 ตามผู้พูด และกลุ่มทดสอบการฟัง 30 ผู้พูด กันออกก่อน
ตัดคลิปที่ประโยคมี "ๆ" หรืออักษรภาษาอังกฤษ (`--drop-pattern`) เพราะระบบโคลนข้ามหรืออ่านเพี้ยน
และตัดประโยคที่สั้นกว่า 8 ตัวอักษร (`--min-chars`) เพราะโมเดลตระกูล F5 จะดึงคำจากเสียงต้นแบบมาเติมเวลาที่เหลือ

ผลลัพธ์อยู่ที่ `data/cohort/<run-id>/`
- `items.csv` เสียงจริง เสียงต้นแบบ และเสียงต้นทางสำหรับ VC ของแต่ละรายการ (มี client_id อยู่ในเครื่องเท่านั้น)
- `summary.json` จำนวนต่อชุด ไม่มีรหัสผู้พูดหรือข้อความ

สคริปต์ตรวจว่าผู้พูดไม่ซ้ำข้ามชุด ประโยคไม่ซ้ำข้ามชุด ไม่มีผู้พูดหรือประโยคจาก pilot
และผู้พูดที่ให้เสียงต้นทาง VC ไม่อยู่ในชุดใดเลย

## ขั้น 2: นำร่อง OmniVoice

คลิปนำร่อง 15 คู่มาจากผู้พูดที่ไม่อยู่ใน cohort ชุดไหนเลย ใช้ `data/pilot_items/pilot_items_v2` (ตัดตามเกณฑ์เดียวกับ cohort_v3; v1 เป็นชุดเก่าที่ใช้กับ OmniVoice/F5 รอบแรก)

ติดตั้งครั้งแรก (environment แยก ไม่ปนกับของเดิม):

```powershell
cd D:\KKU_files\year3_semester2\CP353764\thai-cloned-speech-detection
py -3.12 -m venv .venv-omnivoice
.\.venv-omnivoice\Scripts\python.exe -m pip install --upgrade pip
.\.venv-omnivoice\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv-omnivoice\Scripts\python.exe -m pip install omnivoice soundfile pandas --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv-omnivoice\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

บรรทัดสุดท้ายต้องได้ `2.8.0+cu128 True` ถ้าไม่มีคำสั่ง `py` ให้ใช้ path เต็ม
`& "C:\Users\filmz\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv-omnivoice`

ลองก่อน 2 คลิป (ครั้งแรกจะดาวน์โหลดโมเดลจาก Hugging Face ไปไว้ที่ `checkpoints\hf\` ในโปรเจกต์ ไม่ลงไดรฟ์ C):

```powershell
$CV = "data\raw\common_voice\cv-corpus-27.0-2026-09-11\th"
.\.venv-omnivoice\Scripts\python.exe scripts\03_pilot_omnivoice.py --items data\pilot_items\pilot_items_v2\pilot_items.csv --cv-dir $CV --run-id omni_try_01 --limit 2
```

ถ้าผ่าน รันครบ 15 คลิปด้วย run-id ใหม่ (`--run-id omni_pilot_v1` ไม่ใส่ `--limit`) แล้วเปิด
`data\pilot\omnivoice\<run-id>\index.html` ฟังเทียบ เสียงจริง / clone A (ตัดเงียบ) / clone B (ไม่ตัด) / เสียงต้นแบบ
ตัวเลขเวลาและ VRAM อยู่ใน `run.json` และ `clones.csv`

## ขั้น 2: นำร่อง F5-TTS-THAI

ติดตั้งครั้งแรก (environment แยก `.venv-f5` ในโปรเจกต์นี้ ไม่ใช้ของโปรเจกต์เก่า):

```powershell
py -3.12 -m venv .venv-f5
.\.venv-f5\Scripts\python.exe -m pip install --upgrade pip
.\.venv-f5\Scripts\python.exe -m pip install --no-cache-dir torch==2.4.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv-f5\Scripts\python.exe -m pip install --no-cache-dir f5-tts-th==1.0.9 torch==2.4.1+cu124 torchaudio==2.4.1+cu124 soundfile scipy pandas --extra-index-url https://download.pytorch.org/whl/cu124
.\.venv-f5\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

บรรทัดสุดท้ายต้องได้ `2.4.1+cu124 True`

บทเรียนจากโปรเจกต์เก่าบนเครื่องนี้: torch 2.4.0 เปิดไม่ขึ้น (ขาด libomp140) จึงใช้ 2.4.1 และไม่ใช้ torchaudio >= 2.9
เพราะ f5-tts-th เรียก `torchaudio.load` ซึ่งรุ่นใหม่ต้องมี ffmpeg

ลองก่อน 2 คลิป (ครั้งแรกดาวน์โหลดโมเดล ~1.4 GB ไป `checkpoints\hf\`):

```powershell
$CV = "data\raw\common_voice\cv-corpus-27.0-2026-09-11\th"
.\.venv-f5\Scripts\python.exe scripts\04_pilot_f5_thai.py --items data\pilot_items\pilot_items_v2\pilot_items.csv --cv-dir $CV --run-id f5_try_01 --limit 2
```

ผลอยู่ `data\pilot\f5_tts_thai\<run-id>\index.html` แบบ A = speed 1.0 (ค่าเริ่มต้น), แบบ B = กำหนดความยาวเท่าเสียงจริง

## กฎของโฟลเดอร์นี้

- ใช้ run-id ใหม่ทุกครั้ง สคริปต์จะไม่ยอมเขียนทับ
- ห้าม commit เสียง เสียงโคลน เสียงต้นแบบ client_id หรือรายการรายคลิป (`data/` ถูก ignore แล้ว)
- ไม่พยายามระบุตัวผู้พูด และไม่เผยแพร่เสียงโคลน
- ชุด Test เปิดครั้งเดียวหลังล็อกทุกอย่าง ห้ามใช้เลือกค่าใด ๆ
