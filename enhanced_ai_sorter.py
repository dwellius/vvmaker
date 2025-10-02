#!/usr/bin/env python3
"""
Enhanced AI Video Sorter - Улучшенная версия для создания вирусных клипов
Похоже на OpusClip, но с дополнительными возможностями и бесплатным AI анализом
"""

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
install_requirements()

from faster_whisper import WhisperModel
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import openai
import nltk
import textstat
from colorama import init, Fore, Style, Back
from tqdm import tqdm

# Инициализируем цвета для красивого вывода
init(autoreset=True)

# Настройки OpenAI для бесплатного анализа
openai.api_key = "sk-emergent-fDbDaD9E72a58Ab4bE"
openai.base_url = "https://api.emergent.com"

class VideoAnalyzer:
    """Улучшенный анализатор видео с AI поддержкой"""
    
    def __init__(self, config_file="config.json"):
        self.config = self.load_config(config_file)
        self.setup_nltk()
        
    def setup_nltk(self):
        """Настройка NLTK для анализа текста"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Загружаю языковые данные...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
    
    def load_config(self, config_file):
        """Загрузка конфигурации"""
        default_config = {
            "whisper_model": "small",
            "scene_threshold": 30.0,
            "min_clip_duration": 15,
            "max_clip_duration": 60,
            "min_words": 8,
            "output_resolution": "1080x1920",
            "subtitle_styles": {
                "modern": {
                    "font": "Arial Black",
                    "size": 56,
                    "color": "&H00FFFFFF",
                    "outline": 4,
                    "shadow": 2
                },
                "neon": {
                    "font": "Impact", 
                    "size": 64,
                    "color": "&H0000FFFF",
                    "outline": 6,
                    "shadow": 3
                },
                "elegant": {
                    "font": "Times New Roman",
                    "size": 52,
                    "color": "&H00F0F0F0", 
                    "outline": 3,
                    "shadow": 1
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
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
                
        return default_config

class EnhancedTranscriber:
    """Улучшенная система транскрипции с анализом эмоций"""
    
    def __init__(self, model_size="small"):
        self.model = WhisperModel(model_size, device="cpu")
        print(f"{Fore.CYAN}🎤 Whisper модель загружена: {model_size}")
        
    def transcribe_with_analysis(self, video_path: str) -> Dict:
        """Транскрипция с анализом контента"""
        print(f"{Fore.YELLOW}📝 Транскрибирую видео: {video_path}")
        
        segments, info = self.model.transcribe(
            video_path, 
            word_timestamps=True,
            language="ru"  # Оптимизируем для русского языка
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
            
            # Анализ сегмента
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": segment_text,
                "words": words,
                "word_count": len(segment_text.split()),
                "reading_level": textstat.flesch_reading_ease(segment_text) if segment_text else 0,
                "excitement_score": self.calculate_excitement(segment_text),
                "has_question": "?" in segment_text,
                "has_exclamation": "!" in segment_text
            }
            
            transcript.append(segment_data)
        
        return {
            "segments": transcript,
            "full_text": full_text.strip(),
            "total_duration": info.duration,
            "language": info.language,
            "language_probability": info.language_probability
        }
    
    def calculate_excitement(self, text: str) -> float:
        """Вычисляет уровень 'возбуждения' текста"""
        excitement_words = [
            "невероятно", "удивительно", "фантастически", "потрясающе",
            "шок", "вау", "офигеть", "круто", "супер", "мега"
        ]
        
        score = 0
        words = text.lower().split()
        
        for word in words:
            if word in excitement_words:
                score += 2
            if word.endswith("!"):
                score += 1
                
        # Нормализуем по длине текста
        return min(score / max(len(words), 1), 1.0)

class AIContentAnalyzer:
    """AI анализатор контента для определения лучших моментов"""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key="sk-emergent-fDbDaD9E72a58Ab4bE",
            base_url="https://api.emergent.com"
        )
    
    def analyze_segments(self, transcript_data: Dict) -> List[Dict]:
        """Анализ сегментов с помощью AI"""
        segments = transcript_data["segments"]
        
        print(f"{Fore.MAGENTA}🤖 Анализирую контент с помощью AI...")
        
        # Группируем сегменты для анализа (по 5-10 сегментов)
        analyzed_segments = []
        
        for i in range(0, len(segments), 8):
            batch = segments[i:i+8]
            batch_text = "\n".join([f"{j+1}. {seg['text']}" for j, seg in enumerate(batch)])
            
            if len(batch_text.strip()) < 50:  # Слишком короткий текст
                for seg in batch:
                    seg["ai_score"] = 0.3
                    seg["ai_reason"] = "Недостаточно контента"
                analyzed_segments.extend(batch)
                continue
            
            try:
                # Используем бесплатный AI для анализа
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """Ты эксперт по созданию вирусного контента. 
                            Оцени каждый сегмент от 0.0 до 1.0 по потенциалу стать вирусным.
                            Учитывай: эмоциональность, полезность, неожиданность, юмор.
                            Ответь строго в JSON формате:
                            {"segments": [{"index": 1, "score": 0.8, "reason": "причина"}]}"""
                        },
                        {
                            "role": "user",
                            "content": f"Оцени эти сегменты:\n{batch_text}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                result = json.loads(response.choices[0].message.content)
                
                for item in result.get("segments", []):
                    idx = item.get("index", 1) - 1
                    if idx < len(batch):
                        batch[idx]["ai_score"] = item.get("score", 0.5)
                        batch[idx]["ai_reason"] = item.get("reason", "AI анализ")
                        
            except Exception as e:
                print(f"{Fore.RED}⚠️ Ошибка AI анализа: {e}")
                # Fallback анализ без AI
                for seg in batch:
                    score = self.fallback_analysis(seg)
                    seg["ai_score"] = score
                    seg["ai_reason"] = "Локальный анализ"
            
            analyzed_segments.extend(batch)
            time.sleep(0.1)  # Небольшая пауза между запросами
        
        return analyzed_segments
    
    def fallback_analysis(self, segment: Dict) -> float:
        """Резервный анализ без AI"""
        score = 0.0
        text = segment["text"].lower()
        
        # Базовая оценка по ключевым словам
        high_value_words = ["секрет", "ошибка", "совет", "важно", "лайфхак"]
        emotion_words = ["смешно", "удивительно", "шок", "невероятно"]
        
        for word in high_value_words:
            if word in text:
                score += 0.3
                
        for word in emotion_words:
            if word in text:
                score += 0.2
        
        # Добавляем баллы за вопросы и восклицания
        if "?" in text:
            score += 0.1
        if "!" in text:
            score += 0.15
            
        # Учитываем excitement_score
        score += segment.get("excitement_score", 0) * 0.2
        
        return min(score, 1.0)

class AdvancedSceneDetector:
    """Улучшенный детектор сцен"""
    
    def __init__(self, threshold=30.0):
        self.threshold = threshold
        
    def detect_scenes(self, video_path: str) -> List[Tuple]:
        """Обнаружение сцен с улучшенными параметрами"""
        print(f"{Fore.GREEN}🎬 Анализирую сцены в видео...")
        
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        # Используем более чувствительный детектор
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            duration = (end - start).get_seconds()
            scenes.append({
                "start_timecode": start.get_timecode(),
                "end_timecode": end.get_timecode(), 
                "start_seconds": start.get_seconds(),
                "end_seconds": end.get_seconds(),
                "duration": duration,
                "scene_id": i + 1
            })
            
        print(f"{Fore.GREEN}✅ Найдено сцен: {len(scenes)}")
        return scenes

class EnhancedSubtitleGenerator:
    """Улучшенный генератор субтитров с множественными стилями"""
    
    def __init__(self, config):
        self.config = config
        self.styles = config["subtitle_styles"]
        
    def generate_ass_subtitles(self, transcript, start_sec, end_sec, output_path, style_name="modern"):
        """Генерация ASS субтитров с выбранным стилем"""
        
        style = self.styles.get(style_name, self.styles["modern"])
        
        header = f"""[Script Info]
Title: Enhanced AI Clip
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['font']},{style['size']},{style['color']}&HFFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},2,30,30,100,1
Style: Highlight,{style['font']},{style['size'] + 8},&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,110,110,0,0,1,{style['outline'] + 1},{style['shadow'] + 1},2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        lines = [header]
        emoji_map = self.config["emoji_map"]
        
        for segment in transcript:
            if (segment["start"] >= start_sec and segment["end"] <= end_sec and 
                segment.get("words")):
                
                seg_start = segment["start"] - start_sec
                seg_end = segment["end"] - start_sec
                
                # Определяем стиль на основе AI оценки
                use_highlight = segment.get("ai_score", 0) > 0.7
                subtitle_style = "Highlight" if use_highlight else "Default"
                
                # Создаем караоке эффект
                karaoke_text = self.create_karaoke_effect(segment["words"], emoji_map)
                
                if karaoke_text.strip():
                    lines.append(
                        f"Dialogue: 0,{self.format_time(seg_start)},{self.format_time(seg_end)},"
                        f"{subtitle_style},,0,0,0,,{karaoke_text}"
                    )
        
        # Сохраняем файл
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def create_karaoke_effect(self, words, emoji_map):
        """Создание караоке эффекта с эмодзи"""
        karaoke_text = ""
        
        for word_data in words:
            duration_cs = max(int((word_data["end"] - word_data["start"]) * 100), 10)
            word_text = word_data["text"]
            
            # Добавляем эмодзи
            for keyword, emoji in emoji_map.items():
                if keyword.lower() in word_text.lower():
                    word_text = f"{word_text} {emoji}"
                    break
            
            # Добавляем анимационные эффекты для важных слов
            confidence = word_data.get("confidence", 1.0)
            if confidence < 0.7:  # Неуверенное слово
                word_text = f"{{\\alpha&H80&}}{word_text}{{\\alpha&H00&}}"
            
            karaoke_text += f"{{\\kf{duration_cs}}}{word_text} "
        
        return karaoke_text.strip()
    
    def format_time(self, seconds):
        """Форматирование времени для ASS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds * 100) % 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

class AdvancedVideoProcessor:
    """Улучшенный процессор видео с продвинутыми эффектами"""
    
    def __init__(self, config):
        self.config = config
        
    def create_enhanced_clips(self, video_path, analyzed_segments, scenes, output_dir="enhanced_clips"):
        """Создание улучшенных клипов"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"{Fore.CYAN}🎥 Создаю улучшенные клипы...")
        
        # Фильтруем и сортируем сегменты по AI оценке
        good_segments = [s for s in analyzed_segments if s.get("ai_score", 0) > 0.4]
        good_segments.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        
        # Берем топ сегменты
        top_segments = good_segments[:min(10, len(good_segments))]
        
        created_clips = []
        
        for i, segment in enumerate(tqdm(top_segments, desc="Создаю клипы")):
            # Находим подходящую сцену
            scene = self.find_best_scene(segment, scenes)
            if not scene:
                continue
                
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
    
    def create_single_clip(self, video_path, segment, scene, clip_number, output_dir):
        """Создание одного клипа с эффектами"""
        
        output_path = os.path.join(output_dir, f"clip_{clip_number:03d}.mp4")
        ass_path = os.path.join(output_dir, f"clip_{clip_number:03d}.ass")
        
        # Генерируем субтитры
        subtitle_gen = EnhancedSubtitleGenerator(self.config)
        style = self.choose_subtitle_style(segment)
        
        subtitle_gen.generate_ass_subtitles(
            [segment], scene["start_seconds"], scene["end_seconds"],
            ass_path, style
        )
        
        # Создаем видео эффекты
        effects = self.generate_video_effects(segment, scene)
        
        # Команда FFmpeg
        cmd = self.build_ffmpeg_command(
            video_path, scene, ass_path, output_path, effects
        )
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                return {
                    "path": output_path,
                    "segment": segment,
                    "scene": scene,
                    "ai_score": segment.get("ai_score", 0),
                    "duration": scene["duration"]
                }
            else:
                print(f"{Fore.RED}❌ Ошибка создания клипа {clip_number}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"{Fore.RED}⏱️ Таймаут при создании клипа {clip_number}")
            return None
        except Exception as e:
            print(f"{Fore.RED}💥 Ошибка: {e}")
            return None
    
    def choose_subtitle_style(self, segment):
        """Выбор стиля субтитров на основе контента"""
        ai_score = segment.get("ai_score", 0)
        
        if ai_score > 0.8:
            return "neon"  # Для очень важного контента
        elif ai_score > 0.6:
            return "modern"  # Для хорошего контента
        else:
            return "elegant"  # Для обычного контента
    
    def generate_video_effects(self, segment, scene):
        """Генерация видео эффектов"""
        effects = {
            "zoom": random.choice([1.0, 1.05, 1.1, 1.15]),
            "pan": random.choice([None, "left", "right", "up", "down"]),
            "color_grade": random.choice([None, "warm", "cool", "vibrant"]),
            "transition": random.choice(["fade", "slide", "zoom_in"]),
            "jump_cut": random.random() > 0.7,  # 30% вероятность
            "speed_ramp": segment.get("ai_score", 0) > 0.7  # Для важного контента
        }
        
        return effects
    
    def build_ffmpeg_command(self, video_path, scene, ass_path, output_path, effects):
        """Построение команды FFmpeg с эффектами"""
        
        # Базовые параметры
        start_time = scene["start_timecode"]
        duration = scene["duration"]
        
        # Создаем фильтр
        video_filters = []
        
        # 1. Базовое кадрирование и масштабирование
        video_filters.append("[0:v]scale=1080:1920,boxblur=20:5[bg]")
        
        # 2. Передний план
        crop_scale = (
            "[0:v]crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,"
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        )
        
        # 3. Добавляем zoom эффект
        if effects["zoom"] > 1.0:
            zoom_filter = f",zoompan=z='min(zoom+0.002,{effects['zoom']})':d=1:s=1080x1920"
            crop_scale += zoom_filter
        
        # 4. Цветокоррекция
        if effects["color_grade"]:
            if effects["color_grade"] == "warm":
                crop_scale += ",eq=temperature=200:brightness=0.05"
            elif effects["color_grade"] == "cool":
                crop_scale += ",eq=temperature=-200:brightness=0.05"
            elif effects["color_grade"] == "vibrant":
                crop_scale += ",eq=saturation=1.2:contrast=1.1"
        
        crop_scale += "[fg]"
        video_filters.append(crop_scale)
        
        # 5. Композитинг
        composite = f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        
        # 6. Jump cut эффект
        if effects["jump_cut"]:
            cut_time = random.uniform(duration * 0.3, duration * 0.7)
            cut_duration = 0.3
            composite += f",select='not(between(t,{cut_time},{cut_time + cut_duration}))',setpts=N/FRAME_RATE/TB"
        
        # 7. Добавляем субтитры
        composite += f",ass={ass_path}"
        
        video_filters.append(composite)
        
        # Собираем команду
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-ss", start_time,
            "-t", str(duration),
            "-vf", ";".join(video_filters),
            "-c:a", "aac", "-b:a", "128k",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-movflags", "+faststart",
            output_path
        ]
        
        return cmd

class EnhancedAIVideoSorter:
    """Главный класс улучшенного AI сортировщика"""
    
    def __init__(self, config_file="config.json"):
        self.analyzer = VideoAnalyzer(config_file)
        self.config = self.analyzer.config
        
        # Инициализируем компоненты
        self.transcriber = EnhancedTranscriber(self.config["whisper_model"])
        self.ai_analyzer = AIContentAnalyzer()
        self.scene_detector = AdvancedSceneDetector(self.config["scene_threshold"])
        self.video_processor = AdvancedVideoProcessor(self.config)
        
        print(f"{Fore.GREEN}{Style.BRIGHT}🚀 Enhanced AI Video Sorter готов к работе!")
        
    def process_video(self, video_path: str, output_dir: str = "enhanced_clips"):
        """Основной процесс обработки видео"""
        
        if not os.path.exists(video_path):
            print(f"{Fore.RED}❌ Файл не найден: {video_path}")
            return None
        
        print(f"{Fore.CYAN}{Style.BRIGHT}🎬 Начинаю обработку: {video_path}")
        print("=" * 60)
        
        # 1. Транскрипция
        print(f"{Back.BLUE}{Fore.WHITE} ЭТАП 1: ТРАНСКРИПЦИЯ {Style.RESET_ALL}")
        transcript_data = self.transcriber.transcribe_with_analysis(video_path)
        
        # 2. AI анализ
        print(f"\n{Back.MAGENTA}{Fore.WHITE} ЭТАП 2: AI АНАЛИЗ КОНТЕНТА {Style.RESET_ALL}")
        analyzed_segments = self.ai_analyzer.analyze_segments(transcript_data)
        
        # 3. Обнаружение сцен
        print(f"\n{Back.GREEN}{Fore.WHITE} ЭТАП 3: АНАЛИЗ СЦЕН {Style.RESET_ALL}")
        scenes = self.scene_detector.detect_scenes(video_path)
        
        # 4. Создание клипов
        print(f"\n{Back.RED}{Fore.WHITE} ЭТАП 4: СОЗДАНИЕ КЛИПОВ {Style.RESET_ALL}")
        created_clips = self.video_processor.create_enhanced_clips(
            video_path, analyzed_segments, scenes, output_dir
        )
        
        # 5. Финальный отчет
        self.print_results(created_clips, output_dir)
        
        return created_clips
    
    def print_results(self, clips, output_dir):
        """Печать результатов"""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}=" * 60)
        print(f"🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"📁 Директория: {output_dir}")
        print(f"🎬 Создано клипов: {len(clips)}")
        
        if clips:
            print(f"\n{Fore.YELLOW}📊 ТОП КЛИПЫ ПО AI ОЦЕНКЕ:")
            for i, clip in enumerate(clips[:5], 1):
                score = clip["ai_score"]
                duration = clip["duration"]
                print(f"  {i}. {os.path.basename(clip['path'])} "
                      f"(Оценка: {score:.2f}, Длительность: {duration:.1f}с)")
        
        print(f"\n{Fore.CYAN}💡 Совет: Проверьте клипы и выберите лучшие для публикации!")


def main():
    """Главная функция"""
    print(f"{Fore.MAGENTA}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════╗")
    print("║        🤖 ENHANCED AI VIDEO SORTER 2.0      ║")
    print("║              Powered by AI                   ║")
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
    sorter = EnhancedAIVideoSorter()
    
    # Обрабатываем видео
    clips = sorter.process_video(video_file)
    
    if clips:
        print(f"\n{Fore.GREEN}✅ Готово! Найдите ваши клипы в папке 'enhanced_clips'")
    else:
        print(f"\n{Fore.RED}❌ Не удалось создать клипы. Проверьте входное видео.")


if __name__ == "__main__":
    main()