import os

os.environ['QT_API'] = 'pyside6'

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QFileDialog, QPushButton, QLineEdit, QCheckBox
)
from scipy.io import wavfile
from scipy.signal import spectrogram
import csv


class VisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LipsSync Visualizer")
        self.resize(1200, 800)  # Set the initial size of the main window

        # Create the main widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Create a layout
        layout = QVBoxLayout(self.main_widget)

        # Add a line edit to display the selected folder path
        self.folder_path_edit = QLineEdit(self)
        self.folder_path_edit.setReadOnly(True)
        layout.addWidget(self.folder_path_edit)

        # Add a button to select folder
        self.select_folder_button = QPushButton("Select Folder", self)
        self.select_folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_button)

        # Add checkboxes to select attributes to visualize
        self.jaw_open_checkbox = QCheckBox("jawOpen", self)
        self.mouth_close_checkbox = QCheckBox("mouthClose", self)
        self.lips_distance_checkbox = QCheckBox("LipsDistance", self)
        self.jaw_open_diff_checkbox = QCheckBox("jawOpen - mouthClose", self)
        self.jaw_open_corr_checkbox = QCheckBox("jawOpen * (1 - mouthClose)", self)
        layout.addWidget(self.jaw_open_checkbox)
        layout.addWidget(self.mouth_close_checkbox)
        layout.addWidget(self.lips_distance_checkbox)
        layout.addWidget(self.jaw_open_diff_checkbox)
        layout.addWidget(self.jaw_open_corr_checkbox)

        # Add a button to visualize the selected attributes
        self.visualize_button = QPushButton("Visualize", self)
        self.visualize_button.setEnabled(False)  # Initially disable the visualize button
        self.visualize_button.clicked.connect(self.visualize)
        layout.addWidget(self.visualize_button)

        # Add a button to export the figure
        self.export_button = QPushButton("Export Figure", self)
        self.export_button.setEnabled(False)  # Initially disable the export button
        self.export_button.clicked.connect(self.export_figure)
        layout.addWidget(self.export_button)

        # Placeholder for plots
        self.plot_widget = QWidget(self)
        layout.addWidget(self.plot_widget)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            csv_file = os.path.join(folder_path, "mouth_data.csv")
            audio_file = os.path.join(folder_path, "audio.wav")
            self.load_data(csv_file, audio_file)
            self.visualize_button.setEnabled(True)  # Enable the visualize button

    def load_data(self, csv_file, audio_file):
        # Read the CSV file
        self.time, self.jaw_open, self.mouth_close, self.lips_distance = self.read_csv_file(csv_file)

        # Read the audio file
        self.sampling_rate, self.audio_data = wavfile.read(audio_file)

    def visualize(self):
        self.create_plots()
        self.export_button.setEnabled(True)  # Enable the export button after the figure is created

    def read_csv_file(self, csv_file):
        time = []
        jaw_open = []
        mouth_close = []
        lips_distance = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                time.append(float(row['TimeStamp']))
                jaw_open.append(float(row['jawOpen']))
                mouth_close.append(float(row['mouthClose']))
                lips_distance.append(float(row['LipsDistance']))
        return np.array(time), np.array(jaw_open), np.array(mouth_close), np.array(lips_distance)

    def create_plots(self):
        # Create a figure
        self.fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size

        # Plot the spectrogram
        f, t, Sxx = spectrogram(
            self.audio_data, self.sampling_rate,
            window='hann', nperseg=2048, noverlap=1536, nfft=2048
        )
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_title("Spectrogram with selected attributes")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim([0, t[-1]])  # Align x-axis with other plots

        # Initialize ax2 to None
        ax2 = None

        # Plot the selected attributes on the secondary y-axis if any are checked
        if self.jaw_open_checkbox.isChecked():
            if ax2 is None:
                ax2 = ax.twinx()
            ax2.plot(self.time, self.jaw_open, 'r-', label='jawOpen')
        if self.mouth_close_checkbox.isChecked():
            if ax2 is None:
                ax2 = ax.twinx()
            ax2.plot(self.time, self.mouth_close, 'b-', label='mouthClose')
        if self.jaw_open_diff_checkbox.isChecked():
            jaw_open_diff = self.jaw_open - self.mouth_close
            if ax2 is None:
                ax2 = ax.twinx()
            ax2.plot(self.time, jaw_open_diff, 'g-', label='jawOpen - mouthClose')
        if self.jaw_open_corr_checkbox.isChecked():
            jaw_open_corr = self.jaw_open * (1 - self.mouth_close)
            if ax2 is None:
                ax2 = ax.twinx()
            ax2.plot(self.time, jaw_open_corr, 'm-', label='jawOpen * (1 - mouthClose)')

        if ax2 is not None:
            ax2.set_ylabel("Attribute Value", color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            ax2.legend(loc='upper right')

        # Plot the LipsDistance attribute on a third y-axis if checked
        if self.lips_distance_checkbox.isChecked():
            ax3 = ax.twinx()
            if ax2 is not None:
                ax3.spines['right'].set_position(('outward', 40))  # Offset the third y-axis if ax2 is rendered
            ax3.plot(self.time, self.lips_distance, 'c-', label='LipsDistance [cm]')
            ax3.set_ylabel("LipsDistance [cm]", color='c')
            ax3.tick_params(axis='y', labelcolor='c')
            ax3.legend(loc='upper left')

        # Add the figure to the layout
        canvas = FigureCanvas(self.fig)
        layout = self.plot_widget.layout()
        if layout is None:
            layout = QVBoxLayout(self.plot_widget)
        else:
            # Clear previous plots
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        layout.addWidget(canvas)

    def export_figure(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_path:
            self.fig.savefig(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationApp()
    window.show()
    sys.exit(app.exec())
