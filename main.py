import os

os.environ['QT_API'] = 'pyside6'

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QPushButton
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

        # Add a button to select folder
        self.select_folder_button = QPushButton("Select Folder", self)
        self.select_folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_button)

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
            csv_file = os.path.join(folder_path, "mouth_data.csv")
            audio_file = os.path.join(folder_path, "audio.wav")
            self.load_and_plot_data(csv_file, audio_file)

    def load_and_plot_data(self, csv_file, audio_file):
        # Read the CSV file
        self.time, self.jaw_open, self.mouth_close = self.read_csv_file(csv_file)

        # Read the audio file
        self.sampling_rate, self.audio_data = wavfile.read(audio_file)

        # Create and add the plots
        self.create_plots()

        # Enable the export button after the figure is created
        self.export_button.setEnabled(True)

    def read_csv_file(self, csv_file):
        time = []
        jaw_open = []
        mouth_close = []
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                time.append(float(row['TimeStamp']))
                jaw_open.append(float(row['jawOpen']))
                mouth_close.append(float(row['mouthClose']))
        return np.array(time), np.array(jaw_open), np.array(mouth_close)

    def create_plots(self):
        # Create a figure
        self.fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
        self.fig.tight_layout(pad=4.0)  # Add padding

        # Plot the spectrogram
        f, t, Sxx = spectrogram(self.audio_data, self.sampling_rate)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_title("Spectrogram with jawOpen and mouthClose")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim([0, t[-1]])  # Align x-axis with other plots

        # Plot the jawOpen attribute on the secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(self.time, self.jaw_open, 'r-', label='jawOpen')
        ax2.set_ylabel("jawOpen Value", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Plot the mouthClose attribute on the same secondary y-axis
        ax2.plot(self.time, self.mouth_close, 'b-', label='mouthClose')
        ax2.set_ylabel("Attribute Value", color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Calculate and plot the new attributes
        jaw_open_minus_mouth_close = self.jaw_open - self.mouth_close
        jaw_open_times_one_minus_mouth_close = self.jaw_open * (1 - self.mouth_close)

        ax2.plot(self.time, jaw_open_minus_mouth_close, 'g-', label='jawOpen - mouthClose')
        ax2.plot(self.time, jaw_open_times_one_minus_mouth_close, 'm-', label='jawOpen * (1 - mouthClose)')

        # Add legends
        ax2.legend(loc='upper right')

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
