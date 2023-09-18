import cv2
#import imageio
import mediapipe as mp
import numpy as np
import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton, QFrame, QStyledItemDelegate
from PyQt5.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt5.QtCore import QTimer, Qt
from sklearn.svm import LinearSVR



class HighScore():
	def __init__(self):
		self.high_score = 0



	def obtain_high_score(self):
		return self.high_score



	def set_high_score(self, high_score):
		self.high_score = high_score



class App(QWidget):
	def __init__(self):
		super().__init__()

		self.score_storage = HighScore()
		self.initUI()
		


	def boundary_check(self, pos, img_w, img_h):
		x, y = pos
		x = max(0, min(x, 1920 - img_w))
		y = max(0, min(y, 1080 - img_h))

		return (x, y)



	def calibration(self):
		cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
		cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		mp_face_mesh = mp.solutions.face_mesh

		self.model_x = LinearSVR(max_iter = 10000)
		self.model_y = LinearSVR(max_iter = 10000)
		data_list = []

		frame_template = np.ones((1080, 1920, 3)).astype(np.uint8)

		size = 50
		target_points = [(960, 540), (size, size), (960, size), (1920 - size, size), (1920 - size, 540), (960, 540), 
						 (size, 540), (size, 1080 - size), (960, 1080 - size), (1920 - size, 1080 - size)]

		#frame_list = []
		for i in range(len(target_points)):
			for j in range(75):
				frame = frame_template.copy()
				r = 255 if (j < 50) else 0
				if j >= 25 and j < 50:
					g = 165
				elif j >= 50:
					g = 255
				else:
					g = 0
				b = 0
				cv2.circle(frame, target_points[i], size, (b, g, r), -1)
				#frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				cv2.imshow('Calibration', frame)
				if cv2.waitKey(1) == 27:
					return None

				ret, feed = self.cap.read()
				if ret:
					with mp_face_mesh.FaceMesh(refine_landmarks = True) as face_mesh:
						results = face_mesh.process(cv2.cvtColor(feed, cv2.COLOR_BGR2RGB))

						if results.multi_face_landmarks:
							for landmarks in results.multi_face_landmarks:
								left_eye_center = None
								right_eye_center = None
								left_eye_landmarks = []
								right_eye_landmarks = []

								for idx, landmark in enumerate(landmarks.landmark):
									if idx in self.left_eye_indices:  
										left_eye_landmarks.append((int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0])))
									if idx == 468:
										left_eye_center = (int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0]))
									if idx in self.right_eye_indices: 
										right_eye_landmarks.append((int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0])))
									if idx == 473:
										right_eye_center = (int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0]))

								left_eye_distances = [self.euclidean_distance(pt, left_eye_center) for pt in left_eye_landmarks]
								right_eye_distances = [self.euclidean_distance(pt, right_eye_center) for pt in right_eye_landmarks]

								feature = left_eye_distances + right_eye_distances + [target_points[i][0], target_points[i][1]]
						else:
							feature = [0] * 114

						data_list.append(feature)
				else:
					continue

			if i != len(target_points) - 1:
				for _ in range(50):
					frame = frame_template.copy()
					cv2.arrowedLine(frame, target_points[i], target_points[i + 1], (255, 255, 255), 10)
					#frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
					cv2.imshow('Calibration', frame)
					if cv2.waitKey(1) == 27:
							break

		cv2.destroyAllWindows()
		data_array = np.vstack(data_list)
		X = data_array[:, : -2]
		y_x = data_array[:, -2]
		y_y = data_array[:, -1]

		self.model_x.fit(X, y_x)
		self.model_y.fit(X, y_y)

		self.btn_game.setEnabled(True)

		#imageio.mimsave('Calib_Demo.gif', frame_list, fps = 20)



	def detect_webcams(self):
		index = 0 
		arr = []
		while True:
			cap = cv2.VideoCapture(index)
			if not cap.read()[0]:
				break
			else:
				arr.append(index)
			cap.release()
			index += 1
		self.combo_webcams.addItems([str(i) for i in arr])



	def direction_vector(self, source, target):
		direction = np.array(target) - np.array(source)
		magnitude = np.linalg.norm(direction)

		if magnitude == 0:  
			return np.array([0, 0])

		return direction / magnitude



	def euclidean_distance(self, pt1, pt2):
		return np.linalg.norm(np.array(pt1) - np.array(pt2))



	def initUI(self):
		self.setWindowTitle('Simple Webcam Gaze Tracking Tech Demo Game')
		self.setWindowIcon(QIcon('bg.png'))

		self.layout = QVBoxLayout()

		self.title = QLabel("Game: Escape from T-1000 !")
		self.title.setAlignment(Qt.AlignCenter)

		self.score_display = QLabel("High Score for Current Session: " + str(self.score_storage.obtain_high_score()) + ' seconds')
		self.score_display.setAlignment(Qt.AlignCenter)

		self.dropdown_header = QLabel("Select Webcam from the Dropdown Menu Below:")
		self.dropdown_header.setAlignment(Qt.AlignCenter)

		self.combo_webcams = QComboBox()

		self.webcam_frame = QLabel(self)
		self.webcam_frame.setFixedSize(640, 480)

		self.btn_calibration = QPushButton('Calibration')
		self.btn_game = QPushButton('Start Game')
		self.btn_game.setEnabled(False)

		self.layout.addWidget(self.title)
		self.layout.addWidget(self.score_display)
		self.layout.addWidget(self.dropdown_header)
		self.layout.addWidget(self.combo_webcams)
		self.layout.addWidget(self.webcam_frame)
		self.layout.addWidget(self.btn_calibration)
		self.layout.addWidget(self.btn_game)
		self.setLayout(self.layout)

		self.combo_webcams.currentIndexChanged.connect(self.select_webcam)
		self.btn_calibration.clicked.connect(self.calibration)
		self.btn_game.clicked.connect(self.play)

		self.timer = QTimer(self)
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(20)

		self.cap = None
		self.calibrated = False
		self.detect_webcams()

		self.left_eye_indices = [466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362, 467, 260, 259, 257, 258, 286, 414, 
								 359, 255, 339, 254, 253, 252, 256, 341, 463, 342, 445, 444, 443, 442, 441, 413, 446, 261, 448, 449, 450, 451, 452, 
								 453, 464, 372, 340, 346, 347, 348, 349, 350, 357, 465]
		self.right_eye_indices = [246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 247, 30, 29, 27, 28, 56, 190, 130, 
								  25, 110, 24, 23, 22, 26, 112, 243, 113, 225, 224, 223, 222, 221, 189, 226, 31, 228, 229, 230, 231, 232, 233, 244, 
								  143, 111, 117, 118, 119, 120, 121, 128, 245]
		self.model_x = None
		self.model_y = None



	def move_towards(self, source, target, step_size):
		direction = self.direction_vector(source, target)
		
		return tuple(np.round(np.array(source) + step_size * direction).astype(int))



	def play(self):
		cv2.namedWindow('Game', cv2.WINDOW_NORMAL)
		cv2.setWindowProperty('Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		mp_face_mesh = mp.solutions.face_mesh

		frame_template = np.ones((1080, 1920, 3)).astype(np.uint8)
		size = 100
		jc_step_size = 10
		t1000_step_size = 8

		jc_img = cv2.resize(cv2.imread('jc.JPG'), (size, size))
		t1000_img = cv2.resize(cv2.imread('t-1000.JPG'), (size, size))

		jc_pos = (960, 540)
		while True:
			t1000_pos = (np.random.randint(size + 1, 1920 - size), np.random.randint(size + 1, 1080 - size))
			if self.euclidean_distance(jc_pos, t1000_pos) > (5 * size * np.sqrt(2)):
				break


		#frame_list = []
		start_time = time.perf_counter()
		while True:
			ret, feed = self.cap.read()
			if ret:
				frame = frame_template.copy()
				with mp_face_mesh.FaceMesh(refine_landmarks = True) as face_mesh:
					results = face_mesh.process(cv2.cvtColor(feed, cv2.COLOR_BGR2RGB))

					if results.multi_face_landmarks:
						for landmarks in results.multi_face_landmarks:
							left_eye_center = None
							right_eye_center = None
							left_eye_landmarks = []
							right_eye_landmarks = []

							for idx, landmark in enumerate(landmarks.landmark):
								if idx in self.left_eye_indices:  
									left_eye_landmarks.append((int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0])))
								if idx == 468:
									left_eye_center = (int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0]))
								if idx in self.right_eye_indices: 
									right_eye_landmarks.append((int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0])))
								if idx == 473:
									right_eye_center = (int(landmark.x * feed.shape[1]), int(landmark.y * feed.shape[0]))

							left_eye_distances = [self.euclidean_distance(pt, left_eye_center) for pt in left_eye_landmarks]
							right_eye_distances = [self.euclidean_distance(pt, right_eye_center) for pt in right_eye_landmarks]

							feature = left_eye_distances + right_eye_distances
					else:
						feature = [0] * 112

				x_pred = int(self.model_x.predict(np.array(feature).reshape(1, -1)))
				y_pred = int(self.model_y.predict(np.array(feature).reshape(1, -1)))
				target_pos = (x_pred, y_pred)
				jc_pos = self.boundary_check(self.move_towards(jc_pos, target_pos, jc_step_size), size, size)
				t1000_pos = self.boundary_check(self.move_towards(t1000_pos, jc_pos, t1000_step_size), size, size)

				end_time = time.perf_counter()
				elapsed_time = end_time - start_time
				elapsed_time_str = "{:.2f}".format(elapsed_time)

				if self.euclidean_distance(jc_pos, t1000_pos) < (size * np.sqrt(2)):
					frame = frame_template.copy()
					if elapsed_time > self.score_storage.obtain_high_score():
						self.score_storage.set_high_score(elapsed_time)
						self.score_display.setText("High Score for Current Session: " + elapsed_time_str + ' s')
					cv2.putText(frame, 'After running for ' + elapsed_time_str + ' seconds, John Connor is caught by the T-1000!', 
								(100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
					cv2.putText(frame, 'Game Over!', (600, 550), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
					for _ in range(100):
						#frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
						cv2.imshow('Game', frame)
						cv2.waitKey(1)
					break

				frame[jc_pos[1] : jc_pos[1] + size, jc_pos[0] : jc_pos[0] + size] = jc_img
				frame[t1000_pos[1] : t1000_pos[1] + size, t1000_pos[0] : t1000_pos[0] + size] = t1000_img

				cv2.putText(frame, 'Time: ' + elapsed_time_str + ' seconds', (1250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
				#frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				cv2.imshow('Game', frame)
				if cv2.waitKey(1) == 27:
					break

		cv2.destroyAllWindows()
		#imageio.mimsave('Game_Demo.gif', frame_list, fps = 10)



	def select_webcam(self):
		if self.cap:
			self.cap.release()

		cam_id = int(self.combo_webcams.currentText())
		self.cap = cv2.VideoCapture(cam_id)



	def update_frame(self):
		if self.cap:
			ret, frame = self.cap.read()
			if ret:
				rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				h, w, ch = rgb_image.shape
				bytes_per_line = ch * w
				self.qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
				self.webcam_frame.setPixmap(QPixmap.fromImage(self.qt_image).scaled(640, 480, Qt.KeepAspectRatio))



def main():
	app = QApplication(sys.argv)
	ex = App()
	ex.setFixedSize(670, 720)
	ex.show()
	sys.exit(app.exec_())



if __name__ == '__main__':
	main()