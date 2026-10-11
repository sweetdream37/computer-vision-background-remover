"""Вариант 7: выделение объекта с фона. Python 3.9 / PySide6."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QRect, Signal, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QVBoxLayout,
    QSpinBox, QFormLayout, QGroupBox, QScrollArea)


def segment(rgb, rect, iterations=5, blur=5, morphology=3):
    """GrabCut → Gaussian blur маски → порог → открытие/закрытие → контуры.

    Возвращает RGBA, бинарную маску, контуры на RGB и время в секундах.
    Размытие применяется к маске; исходные цвета объекта сохраняются.
    """
    started = time.perf_counter()
    h, w = rgb.shape[:2]
    x, y, rw, rh = map(int, rect)
    if not (0 <= x < w and 0 <= y < h and rw >= 2 and rh >= 2
            and x + rw <= w and y + rh <= h and rw * rh < w * h):
        raise ValueError('Выделите прямоугольник внутри фото, оставив фон за его пределами.')
    mask = np.zeros((h, w), np.uint8)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, (x, y, rw, rh),
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iterations, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if blur > 1:
        if blur % 2 == 0:
            raise ValueError('Размер Gaussian blur должен быть нечётным.')
        binary = cv2.GaussianBlur(binary, (blur, blur), 0)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    if morphology > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology, morphology))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = rgb.copy()
    cv2.drawContours(outlined, contours, -1, (255, 70, 40), 2)
    rgba = np.dstack((rgb, binary))
    return rgba, binary, outlined, time.perf_counter() - started


class ImageView(QWidget):
    selectionChanged = Signal(object)

    def __init__(self, selectable=False):
        super().__init__()
        self.selectable = selectable
        self.pixmap = None
        self.image_size = None
        self.selection_rect = None
        self.start = None
        self.setMinimumSize(280, 240)

    def set_array(self, array):
        array = np.ascontiguousarray(array)
        if array.ndim == 2:
            fmt = QImage.Format_Grayscale8
        else:
            fmt = QImage.Format_RGBA8888 if array.shape[2] == 4 else QImage.Format_RGB888
        self.pixmap = QPixmap.fromImage(QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy())
        self.image_size = (array.shape[1], array.shape[0])
        self.selection_rect = None
        self.update()

    def display_rect(self):
        if not self.pixmap:
            return QRect()
        size = self.pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRect((self.width()-size.width())//2, (self.height()-size.height())//2, size.width(), size.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect_or_widget(), Qt.white)
        area = self.display_rect()
        if self.pixmap:
            p.save()
            p.setClipRect(area)
            for y in range(area.top(), area.bottom()+1, 16):
                for x in range(area.left(), area.right()+1, 16):
                    if ((x-area.left())//16 + (y-area.top())//16) % 2 == 0:
                        p.fillRect(x, y, 16, 16, Qt.lightGray)
            p.restore()
            p.drawPixmap(area, self.pixmap)
            if self.selection_rect:
                iw, ih = self.image_size
                x, y, w, h = self.selection_rect
                p.setPen(QPen(Qt.green, 2))
                p.drawRect(QRect(area.x()+round(x*area.width()/iw), area.y()+round(y*area.height()/ih),
                                 round(w*area.width()/iw), round(h*area.height()/ih)))
        else:
            p.drawText(self.rect_or_widget(), Qt.AlignCenter, 'Нет изображения')

    def rect_or_widget(self):
        return super().rect()

    def image_point(self, point):
        area = self.display_rect()
        iw, ih = self.image_size
        return (max(0, min(iw-1, int((point.x()-area.x())*iw/area.width()))),
                max(0, min(ih-1, int((point.y()-area.y())*ih/area.height()))))

    def mousePressEvent(self, event):
        if self.selectable and self.pixmap and event.button() == Qt.LeftButton and self.display_rect().contains(event.position().toPoint()):
            self.start = self.image_point(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self.start is not None:
            end = self.image_point(event.position().toPoint())
            self.selection_rect = (min(self.start[0], end[0]), min(self.start[1], end[1]),
                         abs(end[0]-self.start[0])+1, abs(end[1]-self.start[1])+1)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is not None:
            self.mouseMoveEvent(event)
            self.start = None
            self.selectionChanged.emit(self.selection_rect)


class Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, rgb, rect, settings):
        super().__init__()
        self.rgb, self.rect, self.settings = rgb, rect, settings

    def run(self):
        try:
            self.done.emit(segment(self.rgb, self.rect, *self.settings))
        except Exception as exc:
            self.failed.emit(str(exc))


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Вариант 7 — выделение объекта с фона')
        self.resize(1200, 740)
        self.rgb = self.rgba = self.mask = None
        self.worker = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        buttons = QHBoxLayout()
        self.load = QPushButton('Загрузить фото')
        self.process = QPushButton('Удалить фон')
        self.save_result = QPushButton('Сохранить PNG')
        self.save_mask = QPushButton('Сохранить маску')
        for b in (self.load, self.process, self.save_result, self.save_mask):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        settings = QFormLayout()
        self.iterations = QSpinBox(); self.iterations.setRange(1, 20); self.iterations.setValue(5)
        self.blur = QSpinBox(); self.blur.setRange(1, 31); self.blur.setSingleStep(2); self.blur.setValue(5)
        self.morph = QSpinBox(); self.morph.setRange(1, 21); self.morph.setSingleStep(2); self.morph.setValue(3)
        settings.addRow('Итерации GrabCut', self.iterations)
        settings.addRow('Gaussian blur маски (нечётное; 1 = выкл.)', self.blur)
        settings.addRow('Морфология: открытие и закрытие (1 = выкл.)', self.morph)
        layout.addLayout(settings)
        row = QHBoxLayout()
        self.original = ImageView(True)
        self.mask_view = ImageView()
        self.result = ImageView()
        self.contours = ImageView()
        for title, view in [('Исходное фото — выделите объект мышью', self.original), ('Бинарная маска', self.mask_view), ('Результат с прозрачностью', self.result), ('Контуры объекта', self.contours)]:
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box); box_layout.addWidget(view)
            row.addWidget(box)
        layout.addLayout(row, 1)
        self.status = QLabel('Загрузите фото и обведите объект прямоугольником, оставив немного фона вокруг.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.load.clicked.connect(self.open_photo)
        self.process.clicked.connect(self.run_processing)
        self.save_result.clicked.connect(lambda: self.save(self.rgba, 'object.png'))
        self.save_mask.clicked.connect(lambda: self.save(self.mask, 'mask.png'))
        self.original.selectionChanged.connect(self.selected)
        self.refresh()

    def refresh(self):
        busy = self.worker is not None
        self.load.setEnabled(not busy)
        self.process.setEnabled(not busy and self.rgb is not None and self.original.selection_rect is not None)
        self.original.selectable = not busy
        self.save_result.setEnabled(not busy and self.rgba is not None)
        self.save_mask.setEnabled(not busy and self.mask is not None)

    def selected(self, rect):
        self.status.setText('Выделение: x={}, y={}, ширина={}, высота={}. Нажмите «Удалить фон».'.format(*rect))
        self.refresh()

    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Открыть фото', '', 'Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)')
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
            if min(self.rgb.shape[:2]) < 3:
                raise ValueError('Изображение слишком маленькое.')
            self.original.set_array(self.rgb)
            self.rgba = self.mask = None
            for view in (self.result, self.mask_view, self.contours):
                view.pixmap = None; view.update()
            self.status.setText('Фото загружено. Обведите весь объект с небольшим запасом; фон должен остаться снаружи.')
        except Exception as exc:
            self.rgb = None
            QMessageBox.warning(self, 'Ошибка загрузки', str(exc))
        self.refresh()

    def run_processing(self):
        if self.blur.value() % 2 == 0:
            QMessageBox.warning(self, 'Параметры', 'Размер Gaussian blur должен быть нечётным.')
            return
        self.worker = Worker(self.rgb.copy(), self.original.selection_rect, (self.iterations.value(), self.blur.value(), self.morph.value()))
        self.worker.done.connect(self.completed)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.status.setText('Обработка… На больших фотографиях это может занять некоторое время.')
        self.refresh()
        self.worker.start()

    def completed(self, data):
        self.rgba, self.mask, contours, elapsed = data
        self.result.set_array(self.rgba); self.mask_view.set_array(self.mask); self.contours.set_array(contours)
        self.status.setText('Время обработки: {:.3f} с. Белое в маске — объект, чёрное — фон. {}'.format(elapsed,
            'Объект не найден: попробуйте другое выделение.' if not np.any(self.mask) else 'Можно сохранить PNG и маску.'))

    def failed(self, message):
        self.status.setText('Обработка не выполнена. Измените выделение или параметры.')
        QMessageBox.warning(self, 'Ошибка обработки', message)

    def finished(self):
        self.worker.deleteLater(); self.worker = None; self.refresh()

    def save(self, array, default):
        if array is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить', default, 'PNG (*.png)')
        if path:
            if not path.lower().endswith('.png'):
                path += '.png'
            try:
                Image.fromarray(array).save(path, format='PNG')
                self.status.setText('Сохранено: ' + path)
            except Exception as exc:
                QMessageBox.warning(self, 'Ошибка сохранения', str(exc))

    def closeEvent(self, event):
        if self.worker is not None:
            QMessageBox.information(self, 'Обработка', 'Дождитесь завершения обработки перед закрытием окна.')
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = Window()
    window.show()
    sys.exit(app.exec())
