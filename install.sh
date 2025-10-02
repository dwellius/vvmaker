#!/bin/bash
# Скрипт установки Enhanced AI Video Sorter

echo "🤖 Enhanced AI Video Sorter - Установка"
echo "======================================"

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден! Установите Python 3.8+ и повторите."
    exit 1
fi

echo "✅ Python найден: $(python3 --version)"

# Проверяем pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip не найден! Установите pip и повторите."
    exit 1
fi

echo "✅ pip найден"

# Проверяем FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg не найден!"
    echo "📋 Для установки FFmpeg:"
    echo "   Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
    echo "   macOS: brew install ffmpeg"  
    echo "   Windows: скачайте с https://ffmpeg.org"
    echo ""
    read -p "Продолжить установку без FFmpeg? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Установка отменена"
        exit 1
    fi
else
    echo "✅ FFmpeg найден: $(ffmpeg -version | head -1)"
fi

echo ""
echo "📦 Устанавливаю Python зависимости..."

# Устанавливаем зависимости
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Все зависимости установлены!"
else
    echo "❌ Ошибка установки зависимостей"
    exit 1
fi

echo ""
echo "🧪 Запускаю тест системы..."

# Запускаем тест
python3 test_sorter.py

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "🚀 Для использования:"
echo "   python3 enhanced_ai_sorter.py your_video.mp4"
echo "   python3 run_sorter.py your_video.mp4"
echo ""
echo "📖 Подробные инструкции в README.md"