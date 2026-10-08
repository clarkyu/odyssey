# 塞壬之歌 · 中英双语对白视频 — 构建脚本

成片：仓库根目录 `The Odyssey - Song of The Sirens - Bilingual Dialogue.mp4`（1080×1920 · 30fps · 54s）
字幕：同名 `.srt`（每条英文一行、中文一行）

## 做法

1. **人声分离**：`separate_vocals.py` 用 UVR 的 MDX-Net 模型（`UVR-MDX-NET-Voc_FT.onnx`，onnxruntime 推理）把配音 MP3 里的配乐剥掉，只留英文人声。停顿处残留配乐约 -71 dB，听不到。
2. **对轴**：按人声停顿和逐词强制对齐（pocketsphinx）定下每句起止时间，见 `render.py` 里的 `CUES`。
3. **画面**：仓库视频原声整条丢弃。15 个镜头按镜头单独放慢（0.63–0.80×），让剪辑点落在台词停顿里，39.6s 画面刚好铺满 54s。
4. **字幕**：英文 Cormorant Garamond，中文 Noto Serif SC；说话人切换处标注角色名。

## 复现

```bash
pip install onnxruntime numpy soundfile pillow
# 模型：github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx
# 字体放进 tools/sirens_song/fonts/（或用 FONT_DIR 指定）：
#   CormorantGaramond[wght].ttf、NotoSerifSC[wght].ttf（google/fonts 仓库 ofl/ 目录）

ffmpeg -i dialogue.mp3 -map 0:a -ar 44100 -ac 2 -c:a pcm_f32le mix.wav
python separate_vocals.py mix.wav vocals.wav UVR-MDX-NET-Voc_FT.onnx 1.021
ffmpeg -i vocals.wav -af "highpass=f=60,atrim=0:54,volume=-4.66dB,afade=t=out:st=53.2:d=0.8,aresample=48000" -c:a pcm_s24le voice.wav
python render.py --src "../../The Odyssey - Song of The Sirens - watermark removed.mp4" --audio voice.wav --out out.mp4
# 只出样张：加 --stills 2,15,38.5 （--out 作为 PNG 文件名前缀）
```
