#!/usr/bin/env python3
"""
Быстрый запуск Enhanced AI Video Sorter
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Enhanced AI Video Sorter - создание вирусных клипов')
    parser.add_argument('video', help='Путь к видео файлу (MP4)')
    parser.add_argument('-o', '--output', default='enhanced_clips', help='Папка для вывода')
    parser.add_argument('-c', '--config', default='config.json', help='Файл конфигурации')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                      default='small', help='Размер Whisper модели')
    
    args = parser.parse_args()
    
    # Проверяем существование видео файла
    if not os.path.exists(args.video):
        print(f"❌ Видео файл не найден: {args.video}")
        sys.exit(1)
    
    # Проверяем расширение
    if not args.video.lower().endswith('.mp4'):
        print("⚠️ Рекомендуется использовать MP4 файлы для лучшей совместимости")
    
    # Импортируем и запускаем
    try:
        from enhanced_ai_sorter import EnhancedAIVideoSorter
        
        print(f"🚀 Запускаю обработку видео: {args.video}")
        print(f"📁 Результаты будут сохранены в: {args.output}")
        print(f"🤖 Используется Whisper модель: {args.model}")
        print("=" * 50)
        
        # Создаем сортировщик
        sorter = EnhancedAIVideoSorter(args.config)
        
        # Обновляем модель если нужно
        sorter.config['whisper_model'] = args.model
        sorter.transcriber = sorter.transcriber.__class__(args.model)
        
        # Запускаем обработку
        clips = sorter.process_video(args.video, args.output)
        
        if clips:
            print(f"\n✅ Успешно создано {len(clips)} клипов!")
            print(f"📂 Проверьте папку: {args.output}")
        else:
            print("\n❌ Не удалось создать клипы")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что установлены все зависимости: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()