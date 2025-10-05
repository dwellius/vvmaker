#!/usr/bin/env python3
import os
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import math

# Проверяем и устанавливаем необходимые пакеты
def install_requirements():
    """Автоматическая установка всех необходимых пакетов"""
    packages = [
        'faster-whisper',
        'scenedetect[opencv]',
        'openai',
        'nltk',
        'textstat',
        'colorama',
        'tqdm'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
        except ImportError:
            print(f"Устанавливаю {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Устанавливаем пакеты если их нет
try:
    install_requirements()
except:
    print("⚠️ Некоторые пакеты не установились, продолжаю с базовой функциональностью...")

from faster_whisper import WhisperModel
try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_NEW = True
except ImportError:
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
        SCENEDETECT_NEW = False
    except ImportError:
        print("⚠️ Scenedetect не установлен, используется базовая нарезка")
        SCENEDETECT_NEW = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI не доступен, используется только локальный анализ")

try:
    import nltk
    import textstat
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("⚠️ Аналитические библиотеки не доступны")

from colorama import init, Fore, Style, Back
from tqdm import tqdm

# Инициализируем цвета для красивого вывода
init(autoreset=True)

class VideoAnalyzer:
    """Улучшенный анализатор видео с fallback режимами"""
    
    def __init__(self, config_file="config.json"):
        self.config = self.load_config(config_file)
        if ANALYSIS_AVAILABLE:
            self.setup_nltk()
        
    def setup_nltk(self):
        """Настройка NLTK для анализа текста"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                print("Загружаю языковые данные...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except:
                print("⚠️ Не удалось загрузить NLTK данные, продолжаю без них")
    
    def load_config(self, config_file):
        """Загрузка конфигурации"""
        default_config = {
            "whisper_model": "small",
            "scene_threshold": 30.0,
            "min_clip_duration": 10,  # Уменьшено для коротких видео
            "max_clip_duration": 60,
            "min_words": 5,  # Уменьшено
            "segment_duration": 30,  # Для случая когда сцены не найдены
            "output_resolution": "1080x1920",
            "subtitle_styles": {
                "modern": {
                    "font": "Arial Black",
                    "size": 56,
                    "color": "&H00FFFFFF",
                    "outline": 4,
                    "shadow": 2
                }
            },
            "emoji_map": {
                "смешно": "😂", "юмор": "😄", "ржака": "🤣",
                "ошибка": "❌", "проблема": "⚠️", "баг": "🐛",
                "важно": "⭐", "внимание": "📢", "ключевое": "🔑",
                "идея": "💡", "мысль": "🧠", "инновация": "🚀",
                "совет": "📌", "лайфхак": "💯", "секрет": "🔐",
                "деньги": "💰", "бизнес": "📈", "успех": "🎯",
                "любовь": "❤️", "семья": "👨‍👩‍👧‍👦", "друзья": "👥",
                "спорт": "⚽", "здоровье": "💪", "фитнес": "🏋️",
                "еда": "🍕", "готовка": "👨‍🍳", "рецепт": "📝",
                "путешествие": "✈️", "приключение": "🗺️", "отдых": "🏖️",
                "технологии": "💻", "AI": "🤖", "будущее": "🔮",
                "образование": "📚", "учеба": "🎓", "знания": "📖"
            },
            "keywords": {
                "high_priority": ["важно", "секрет", "ошибка", "совет", "лайфхак", "внимание"],
                "medium_priority": ["интересно", "смешно", "удивительно", "неожиданно"],
                "emotions": ["радость", "грусть", "злость", "удивление", "страх"],
                "engagement": ["подписывайтесь", "лайк", "комментарий", "поделитесь"]
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except:
                print("⚠️ Ошибка чтения конфига, использую стандартные настройки")
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
                
        return default_config

class EnhancedTranscriber:
    """Улучшенная система транскрипции"""
    
    def __init__(self, model_size="small"):
        self.model = WhisperModel(model_size, device="cpu")
        print(f"{Fore.CYAN}🎤 Whisper модель загружена: {model_size}")
        
    def transcribe_with_analysis(self, video_path: str) -> Dict:
        """Транскрипция с анализом контента"""
        print(f"{Fore.YELLOW}📝 Транскрибирую видео: {video_path}")
        
        segments, info = self.model.transcribe(
            video_path, 
            word_timestamps=True
        )
        
        transcript = []
        full_text = ""
        
        for seg in tqdm(segments, desc="Обрабатываю сегменты"):
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "start": w.start,
                        "end": w.end,
                        "text": w.word.strip(),
                        "confidence": getattr(w, 'probability', 1.0)
                    })
            
            segment_text = seg.text.strip()
            full_text += segment_text + " "
            
            # Базовый анализ сегмента
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": segment_text,
                "words": words,
                "word_count": len(segment_text.split()),
                "excitement_score": self.calculate_excitement(segment_text),
                "has_question": "?" in segment_text,
                "has_exclamation": "!" in segment_text
            }
            
            transcript.append(segment_data)
        
        return {
            "segments": transcript,
            "full_text": full_text.strip(),
            "total_duration": info.duration,
            "language": getattr(info, 'language', 'ru'),
            "language_probability": getattr(info, 'language_probability', 1.0)
        }
    
    def calculate_excitement(self, text: str) -> float:
        """Вычисляет уровень 'возбуждения' текста"""
        excitement_words = [
            "невероятно", "удивительно", "фантастически", "потрясающе",
            "шок", "вау", "офигеть", "круто", "супер", "мега", "классно"
        ]
        
        score = 0
        words = text.lower().split()
        
        for word in words:
            if any(ew in word for ew in excitement_words):
                score += 2
            if word.endswith("!") or "!" in text:
                score += 1
                
        # Нормализуем по длине текста
        return min(score / max(len(words), 1), 1.0)

class AIContentAnalyzer:
    """AI анализатор контента с fallback режимом"""
    
    def __init__(self):
        self.ai_available = False
        if OPENAI_AVAILABLE:
            try:
                self.client = openai.OpenAI(
                    api_key="sk-emergent-fDbDaD9E72a58Ab4bE",
                    base_url="https://api.emergent.com"
                )
                self.ai_available = True
            except:
                print("⚠️ AI анализ недоступен, использую локальный анализ")
    
    def analyze_segments(self, transcript_data: Dict) -> List[Dict]:
        """Анализ сегментов с fallback на локальный анализ"""
        segments = transcript_data["segments"]
        
        print(f"{Fore.MAGENTA}🤖 Анализирую контент...")
        
        analyzed_segments = []
        
        # Пробуем AI анализ только если доступен интернет
        if self.ai_available:
            try:
                # Простая проверка соединения
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    timeout=10
                )
                print(f"{Fore.GREEN}✅ AI анализ доступен")
                return self.ai_analyze_segments(segments)
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ AI недоступен ({str(e)[:50]}...), использую локальный анализ")
        
        # Локальный анализ
        return self.local_analyze_segments(segments)
    
    def ai_analyze_segments(self, segments):
        """AI анализ (когда доступен интернет)"""
        analyzed_segments = []
        
        for i in range(0, len(segments), 5):
            batch = segments[i:i+5]
            batch_text = "\n".join([f"{j+1}. {seg['text']}" for j, seg in enumerate(batch)])
            
            if len(batch_text.strip()) < 20:
                for seg in batch:
                    seg["ai_score"] = 0.3
                    seg["ai_reason"] = "Короткий контент"
                analyzed_segments.extend(batch)
                continue
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """Оцени каждый сегмент от 0.0 до 1.0 по интересности для коротких видео.
                            Ответь строго в JSON: {"segments": [{"index": 1, "score": 0.8, "reason": "причина"}]}"""
                        },
                        {
                            "role": "user",
                            "content": f"Оцени эти сегменты:\n{batch_text}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=300,
                    timeout=15
                )
                
                result = json.loads(response.choices[0].message.content)
                
                for item in result.get("segments", []):
                    idx = item.get("index", 1) - 1
                    if idx < len(batch):
                        batch[idx]["ai_score"] = item.get("score", 0.5)
                        batch[idx]["ai_reason"] = item.get("reason", "AI анализ")
                        
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Переключаюсь на локальный анализ")
                for seg in batch:
                    score = self.local_analysis(seg)
                    seg["ai_score"] = score
                    seg["ai_reason"] = "Локальный анализ"
            
            analyzed_segments.extend(batch)
            time.sleep(0.1)
        
        return analyzed_segments
    
    def local_analyze_segments(self, segments):
        """Локальный анализ без AI"""
        print(f"{Fore.CYAN}🔍 Использую локальный анализ контента")
        
        for segment in tqdm(segments, desc="Анализирую локально"):
            score = self.local_analysis(segment)
            segment["ai_score"] = score
            segment["ai_reason"] = "Локальный анализ"
        
        return segments
    
    def local_analysis(self, segment: Dict) -> float:
        """Локальный анализ сегмента"""
        score = 0.0
        text = segment["text"].lower()
        
        # Базовая оценка по ключевым словам
        high_value_words = ["секрет", "ошибка", "совет", "важно", "лайфхак", "внимание"]
        emotion_words = ["смешно", "удивительно", "шок", "невероятно", "круто", "супер"]
        
        for word in high_value_words:
            if word in text:
                score += 0.3
                
        for word in emotion_words:
            if word in text:
                score += 0.25
        
        # Добавляем баллы за вопросы и восклицания
        if "?" in text:
            score += 0.15
        if "!" in text:
            score += 0.2
            
        # Учитываем excitement_score
        score += segment.get("excitement_score", 0) * 0.3
        
        # Длина сегмента (не слишком короткий, не слишком длинный)
        word_count = segment.get("word_count", 0)
        if 8 <= word_count <= 25:
            score += 0.1
        
        return min(score, 1.0)

class ImprovedSceneDetector:
    """Улучшенный детектор сцен с fallback режимом"""
    
    def __init__(self, threshold=30.0):
        self.threshold = threshold
        
    def detect_scenes(self, video_path: str, transcript_data: Dict) -> List[Tuple]:
        """Обнаружение сцен с fallback на сегменты транскрипции"""
        print(f"{Fore.GREEN}🎬 Анализирую сцены в видео...")
        
        # Пробуем использовать scene detection
        if SCENEDETECT_NEW is not None:
            scenes = self.detect_with_scenedetect(video_path)
            if scenes:
                print(f"{Fore.GREEN}✅ Найдено сцен через детектор: {len(scenes)}")
                return scenes
        
        # Если сцены не найдены, создаем их на основе транскрипции
        print(f"{Fore.YELLOW}⚠️ Детектор сцен не сработал, создаю сцены из транскрипции")
        scenes = self.create_scenes_from_transcript(transcript_data)
        print(f"{Fore.GREEN}✅ Создано сцен из транскрипции: {len(scenes)}")
        return scenes
    
    def detect_with_scenedetect(self, video_path: str):
        """Детекция сцен через scenedetect"""
        try:
            if SCENEDETECT_NEW:
                # Новый API
                scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
            else:
                # Старый API  
                video_manager = VideoManager([video_path])
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=self.threshold))
                video_manager.start()
                scene_manager.detect_scenes(frame_source=video_manager)
                scene_list = scene_manager.get_scene_list()
            
            scenes = []
            for i, (start, end) in enumerate(scene_list):
                if SCENEDETECT_NEW:
                    start_sec = start.get_seconds()
                    end_sec = end.get_seconds()
                    start_tc = start.get_timecode()
                    end_tc = end.get_timecode()
                else:
                    start_sec = start.get_seconds()
                    end_sec = end.get_seconds() 
                    start_tc = start.get_timecode()
                    end_tc = end.get_timecode()
                
                duration = end_sec - start_sec
                scenes.append({
                    "start_timecode": start_tc,
                    "end_timecode": end_tc,
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "duration": duration,
                    "scene_id": i + 1
                })
            
            return scenes if scenes else None
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Ошибка детекции сцен: {e}")
            return None
    
    def create_scenes_from_transcript(self, transcript_data: Dict):
        """Создание сцен на основе транскрипции"""
        segments = transcript_data["segments"]
        total_duration = transcript_data.get("total_duration", 0)
        
        if not segments:
            # Если нет сегментов, делим видео на равные части
            return self.create_equal_scenes(total_duration)
        
        scenes = []
        current_scene_start = 0
        current_scene_duration = 0
        target_duration = 30  # Целевая длина сцены
        
        for i, segment in enumerate(segments):
            segment_duration = segment["end"] - segment["start"]
            current_scene_duration += segment_duration
            
            # Создаем новую сцену если:
            # 1. Достигли целевой длительности
            # 2. Это последний сегмент
            if current_scene_duration >= target_duration or i == len(segments) - 1:
                scene_end = segment["end"]
                
                if current_scene_duration >= 10:  # Минимальная длина сцены
                    scenes.append({
                        "start_timecode": self.seconds_to_timecode(current_scene_start),
                        "end_timecode": self.seconds_to_timecode(scene_end),
                        "start_seconds": current_scene_start,
                        "end_seconds": scene_end,
                        "duration": current_scene_duration,
                        "scene_id": len(scenes) + 1
                    })
                
                current_scene_start = scene_end
                current_scene_duration = 0
        
        return scenes if scenes else self.create_equal_scenes(total_duration)
    
    def create_equal_scenes(self, total_duration):
        """Создание равных сцен если ничего не работает"""
        if total_duration <= 0:
            total_duration = 60  # Предполагаем минуту
        
        scene_duration = 30  # 30-секундные сцены
        scenes = []
        
        start = 0
        scene_id = 1
        
        while start < total_duration:
            end = min(start + scene_duration, total_duration)
            
            scenes.append({
                "start_timecode": self.seconds_to_timecode(start),
                "end_timecode": self.seconds_to_timecode(end),
                "start_seconds": start,
                "end_seconds": end,
                "duration": end - start,
                "scene_id": scene_id
            })
            
            start = end
            scene_id += 1
        
        return scenes
    
    def seconds_to_timecode(self, seconds):
        """Конвертация секунд в таймкод"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

class SimpleVideoProcessor:
    """Упрощенный процессор видео с субтитрами"""
    
    def __init__(self, config):
        self.config = config
        
    def create_clips(self, video_path, analyzed_segments, scenes, output_dir="enhanced_clips"):
        """Создание клипов"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"{Fore.CYAN}🎥 Создаю клипы...")
        
        # Фильтруем и сортируем сегменты
        good_segments = [s for s in analyzed_segments if s.get("ai_score", 0) > 0.3]
        if not good_segments:
            good_segments = analyzed_segments[:5]  # Берем первые 5 если ничего не подошло
            
        good_segments.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        
        created_clips = []
        
        for i, segment in enumerate(tqdm(good_segments[:min(10, len(good_segments))], desc="Создаю клипы")):
            # Находим подходящую сцену
            scene = self.find_best_scene(segment, scenes)
            if not scene:
                # Создаем сцену вокруг сегмента
                scene = self.create_scene_around_segment(segment)
                
            clip_info = self.create_single_clip(
                video_path, segment, scene, i + 1, output_dir
            )
            
            if clip_info:
                created_clips.append(clip_info)
        
        return created_clips
    
    def find_best_scene(self, segment, scenes):
        """Найти лучшую сцену для сегмента"""
        seg_start, seg_end = segment["start"], segment["end"]
        
        for scene in scenes:
            if (scene["start_seconds"] <= seg_start and 
                scene["end_seconds"] >= seg_end and
                scene["duration"] >= self.config["min_clip_duration"] and
                scene["duration"] <= self.config["max_clip_duration"]):
                return scene
        
        return None
    
    def create_scene_around_segment(self, segment):
        """Создать сцену вокруг сегмента"""
        seg_start = segment["start"]
        seg_end = segment["end"]
        
        # Расширяем сцену на 15 секунд в каждую сторону
        scene_start = max(0, seg_start - 15)
        scene_end = seg_end + 15
        duration = scene_end - scene_start
        
        return {
            "start_timecode": self.seconds_to_timecode(scene_start),
            "end_timecode": self.seconds_to_timecode(scene_end),
            "start_seconds": scene_start,
            "end_seconds": scene_end,
            "duration": duration,
            "scene_id": 1
        }
    
    def create_single_clip(self, video_path, segment, scene, clip_number, output_dir):
        """Создание одного клипа с субтитрами"""
        
        output_path = os.path.join(output_dir, f"clip_{clip_number:03d}.mp4")
        ass_path = os.path.join(output_dir, f"clip_{clip_number:03d}.ass")
        
        # Создаем субтитры
        self.create_subtitles(segment, scene, ass_path)
        
        # Команда FFmpeg с субтитрами
        # Исправляем путь к ASS файлу
        ass_path_fixed = ass_path.replace('\\', '/')  # Для Windows
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",  # Больше информации об ошибках
            "-i", video_path,
            "-ss", str(scene["start_seconds"]),
            "-t", str(scene["duration"]),
            "-vf", (
                f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"ass='{ass_path_fixed}'"  # Путь в кавычках
            ),
            "-c:a", "aac", "-b:a", "128k",
            "-c:v", "libx264", "-preset", "fast", "-crf", "25",
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            
            # Проверяем есть ли ошибка с ASS файлом
            if result.returncode != 0 and "libass" in result.stderr:
                print(f"{Fore.YELLOW}⚠️ Проблема с субтитрами, создаю без них...")
                # Создаем видео без субтитров
                cmd_no_subs = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", video_path,
                    "-ss", str(scene["start_seconds"]),
                    "-t", str(scene["duration"]),
                    "-vf", (
                        "scale=1080:1920:force_original_aspect_ratio=decrease,"
                        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
                    ),
                    "-c:a", "aac", "-b:a", "128k",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "25",
                    output_path
                ]
                result = subprocess.run(cmd_no_subs, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return {
                    "path": output_path,
                    "segment": segment,
                    "scene": scene,
                    "ai_score": segment.get("ai_score", 0),
                    "duration": scene["duration"]
                }
            else:
                print(f"{Fore.RED}❌ Ошибка создания клипа {clip_number}")
                if result.stderr:
                    print(f"Ошибка FFmpeg: {result.stderr[:200]}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}💥 Ошибка: {e}")
            return None
    
    def create_subtitles(self, segment, scene, ass_path):
        """Создание ASS субтитров для клипа"""
        
        # Заголовок ASS файла
        header = """[Script Info]
Title: Enhanced AI Clip
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,56,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,30,30,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        lines = [header]
        
        # Добавляем субтитры для сегмента
        if segment["words"]:
            # Караоке субтитры (по словам)
            seg_start_in_clip = max(0, segment["start"] - scene["start_seconds"])
            karaoke_text = ""
            
            for word_data in segment["words"]:
                if (word_data["start"] >= scene["start_seconds"] and 
                    word_data["end"] <= scene["end_seconds"]):
                    
                    duration_cs = max(int((word_data["end"] - word_data["start"]) * 100), 10)
                    word_text = word_data["text"]
                    
                    # Добавляем эмодзи
                    emoji_map = self.config.get("emoji_map", {})
                    for keyword, emoji in emoji_map.items():
                        if keyword.lower() in word_text.lower():
                            word_text = f"{word_text} {emoji}"
                            break
                    
                    karaoke_text += f"{{\\kf{duration_cs}}}{word_text} "
            
            if karaoke_text.strip():
                seg_end_in_clip = min(segment["end"] - scene["start_seconds"], scene["duration"])
                
                lines.append(
                    f"Dialogue: 0,{self.format_time(seg_start_in_clip)},{self.format_time(seg_end_in_clip)},"
                    f"Default,,0,0,0,,{karaoke_text.strip()}"
                )
        else:
            # Простые субтитры (весь текст сразу)
            seg_start_in_clip = max(0, segment["start"] - scene["start_seconds"])
            seg_end_in_clip = min(segment["end"] - scene["start_seconds"], scene["duration"])
            
            subtitle_text = segment["text"]
            
            # Добавляем эмодзи
            emoji_map = self.config.get("emoji_map", {})
            for keyword, emoji in emoji_map.items():
                if keyword.lower() in subtitle_text.lower():
                    subtitle_text = f"{subtitle_text} {emoji}"
            
            lines.append(
                f"Dialogue: 0,{self.format_time(seg_start_in_clip)},{self.format_time(seg_end_in_clip)},"
                f"Default,,0,0,0,,{subtitle_text}"
            )
        
        # Сохраняем файл
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def format_time(self, seconds):
        """Форматирование времени для ASS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds * 100) % 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    
    def seconds_to_timecode(self, seconds):
        """Конвертация секунд в таймкод"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class FixedAIVideoSorter:
    """Исправленная версия AI сортировщика"""
    
    def __init__(self, config_file="config.json"):
        self.analyzer = VideoAnalyzer(config_file)
        self.config = self.analyzer.config
        
        # Инициализируем компоненты
        self.transcriber = EnhancedTranscriber(self.config["whisper_model"])
        self.ai_analyzer = AIContentAnalyzer()
        self.scene_detector = ImprovedSceneDetector(self.config["scene_threshold"])
        self.video_processor = SimpleVideoProcessor(self.config)
        
        print(f"{Fore.GREEN}{Style.BRIGHT}🚀 Enhanced AI Video Sorter (исправленная версия) готов!")
        
    def process_video(self, video_path: str, output_dir: str = "enhanced_clips"):
        """Основной процесс обработки видео"""
        
        if not os.path.exists(video_path):
            print(f"{Fore.RED}❌ Файл не найден: {video_path}")
            return None
        
        print(f"{Fore.CYAN}{Style.BRIGHT}🎬 Начинаю обработку: {video_path}")
        print("=" * 60)
        
        try:
            # 1. Транскрипция
            print(f"{Back.BLUE}{Fore.WHITE} ЭТАП 1: ТРАНСКРИПЦИЯ {Style.RESET_ALL}")
            transcript_data = self.transcriber.transcribe_with_analysis(video_path)
            
            if not transcript_data["segments"]:
                print(f"{Fore.RED}❌ Не удалось получить транскрипцию")
                return None
            
            # 2. AI анализ
            print(f"\n{Back.MAGENTA}{Fore.WHITE} ЭТАП 2: АНАЛИЗ КОНТЕНТА {Style.RESET_ALL}")
            analyzed_segments = self.ai_analyzer.analyze_segments(transcript_data)
            
            # 3. Обнаружение сцен
            print(f"\n{Back.GREEN}{Fore.WHITE} ЭТАП 3: АНАЛИЗ СЦЕН {Style.RESET_ALL}")
            scenes = self.scene_detector.detect_scenes(video_path, transcript_data)
            
            if not scenes:
                print(f"{Fore.RED}❌ Не удалось создать сцены")
                return None
            
            # 4. Создание клипов
            print(f"\n{Back.RED}{Fore.WHITE} ЭТАП 4: СОЗДАНИЕ КЛИПОВ {Style.RESET_ALL}")
            created_clips = self.video_processor.create_clips(
                video_path, analyzed_segments, scenes, output_dir
            )
            
            # 5. Финальный отчет
            self.print_results(created_clips, output_dir)
            
            return created_clips
            
        except Exception as e:
            print(f"{Fore.RED}💥 Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_results(self, clips, output_dir):
        """Печать результатов"""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}=" * 60)
        print(f"🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"📁 Директория: {output_dir}")
        print(f"🎬 Создано клипов: {len(clips)}")
        
        if clips:
            print(f"\n{Fore.YELLOW}📊 ТОП КЛИПЫ ПО ОЦЕНКЕ:")
            for i, clip in enumerate(clips[:5], 1):
                score = clip["ai_score"]
                duration = clip["duration"]
                print(f"  {i}. {os.path.basename(clip['path'])} "
                      f"(Оценка: {score:.2f}, Длительность: {duration:.1f}с)")
        
        print(f"\n{Fore.CYAN}💡 Клипы готовы к просмотру!")

def main():
    """Главная функция"""
    print(f"{Fore.MAGENTA}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════╗")
    print("║     🤖 ENHANCED AI VIDEO SORTER              ║")
    print("║         Исправленная версия v1.1.            ║")
    print("╚══════════════════════════════════════════════╝")
    print(Style.RESET_ALL)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = input(f"{Fore.CYAN}📹 Введите путь к видео файлу: {Style.RESET_ALL}")
    
    if not video_file or not os.path.exists(video_file):
        print(f"{Fore.RED}❌ Файл не найден или не указан!")
        return
    
    # Создаем экземпляр сортировщика
    sorter = FixedAIVideoSorter()
    
    # Обрабатываем видео
    clips = sorter.process_video(video_file)
    
    if clips and len(clips) > 0:
        print(f"\n{Fore.GREEN}✅ Готово! Найдите ваши клипы в папке 'enhanced_clips'")
        print(f"{Fore.CYAN}🎬 Создано {len(clips)} клипов для проверки")
    else:
        print(f"\n{Fore.RED}❌ Не удалось создать клипы. Проверьте входное видео.")

if __name__ == "__main__":
    main()