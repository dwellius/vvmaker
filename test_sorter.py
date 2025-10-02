#!/usr/bin/env python3
"""
Тестовый скрипт для проверки Enhanced AI Video Sorter
Создает демонстрационное видео для тестирования
"""

import os
import subprocess
from pathlib import Path

def create_test_video():
    """Создает тестовое видео для демонстрации"""
    
    print("🎬 Создаю тестовое видео...")
    
    # Команда для создания тестового видео с текстом
    test_video_path = "/app/test_video.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=blue:size=1920x1080:duration=30:rate=25",
        "-f", "lavfi", 
        "-i", "sine=frequency=440:duration=30",
        "-vf", """drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:
                 text='Добро пожаловать в Enhanced AI Video Sorter!
                 
                 Это тестовое видео демонстрирует возможности программы.
                 
                 ВАЖНО: Здесь есть секрет для успеха в социальных сетях!
                 
                 Совет номер один: Создавайте короткий и смешной контент.
                 
                 Лайфхак: Используйте ключевые слова для лучшего анализа.
                 
                 Идея: AI поможет найти лучшие моменты автоматически!':
                 fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:
                 box=1:boxcolor=black@0.5""",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-shortest",
        test_video_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Тестовое видео создано: {test_video_path}")
        return test_video_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка создания видео: {e}")
        return None
    except FileNotFoundError:
        print("❌ FFmpeg не найден! Установите FFmpeg для создания тестового видео.")
        return None

def run_test():
    """Запускает тест программы"""
    
    print("🚀 Тестирование Enhanced AI Video Sorter")
    print("=" * 50)
    
    # Создаем тестовое видео
    video_path = create_test_video()
    if not video_path:
        print("❌ Не удалось создать тестовое видео")
        return
    
    print("\n📝 Запускаю анализ...")
    
    try:
        # Импортируем наш модуль
        from enhanced_ai_sorter import EnhancedAIVideoSorter
        
        # Создаем сортировщик
        sorter = EnhancedAIVideoSorter()
        
        # Обрабатываем видео  
        clips = sorter.process_video(video_path, "test_clips")
        
        if clips:
            print(f"\n🎉 ТЕСТ ПРОЙДЕН!")
            print(f"✅ Создано {len(clips)} клипов")
            print(f"📁 Результаты в папке: test_clips/")
            
            # Показываем информацию о клипах
            for i, clip in enumerate(clips[:3], 1):
                print(f"  {i}. {os.path.basename(clip['path'])} - Оценка: {clip['ai_score']:.2f}")
        else:
            print("❌ Тест не пройден - клипы не созданы")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены")
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
    
    # Очищаем тестовый файл
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"\n🗑️ Тестовое видео удалено")
    except:
        pass

if __name__ == "__main__":
    run_test()