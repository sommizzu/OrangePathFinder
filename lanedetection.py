 # 필요한 라이브러리 임포트
import cv2  # OpenCV, 컴퓨터 비전 처리를 위한 라이브러리
import numpy as np  # NumPy, 수치 계산을 위한 라이브러리
import xhat as hw  # xhat, 하드웨어 제어를 위한 사용자 정의 모듈
import config as cfg  # config, 설정 값을 담은 사용자 정의 모듈

# 카메라 초기화: 첫 번째 연결된 카메라 사용
cap = cv2.VideoCapture(0)

# 이전 프레임에서 감지된 차선들을 저장할 리스트 초기화
prev_frames_lines = []

# 현재 속도를 설정 파일에서 정의된 최소 속도로 초기화
current_speed = cfg.min_speed

# 현재 속도를 증가시키는 함수 정의
def increase_speed(current_speed, max_speed, increment):
    if current_speed < max_speed:
        current_speed += increment
    return min(current_speed, max_speed)

# 차량의 방향을 제어하는 함수 정의
def control_direction(distance):
    global current_speed  # 전역 변수인 현재 속도 사용
    current_speed = increase_speed(current_speed, cfg.max_speed, cfg.speed_increment)
    # 각 거리 구간에 따라 서로 다른 속도와 휠 정렬 값을 적용
    if distance < 500:
        hw.motor_one_speed(current_speed + cfg.wheel_alignment_right)
        hw.motor_two_speed(current_speed + cfg.wheel_alignment_left)
        cfg.wheel = 3
    elif 500 <= distance <= 750:
        hw.motor_one_speed(current_speed + cfg.wheel_alignment_right)
        hw.motor_two_speed(current_speed + cfg.wheel_alignment_left)
        cfg.wheel = 2
    else:
        hw.motor_one_speed(current_speed + cfg.wheel_alignment_right)
        hw.motor_two_speed(current_speed + cfg.wheel_alignment_left)
        cfg.wheel = 1

# 메인 루프: 카메라가 오픈된 동안 계속 실행
while(cap.isOpened()):
    # 카메라에서 프레임 읽음
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 조정
    resized_frame = cv2.resize(frame, (480, 270))
    # 관심 영역(ROI) 설정
    roi = resized_frame[int(27*6.5):250, 0:480]

    # HSV 컬러 공간으로 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 노란색 차선 감지를 위한 색 범위 설정
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    # 색 범위에 따른 마스크 생성
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)    
    # 마스크 적용하여 차선 이미지 생성
    yellow_lane = cv2.bitwise_and(roi, roi, mask=yellow_mask)

    # 차선 감지를 위한 이미지 블러 처리
    blurred = cv2.GaussianBlur(yellow_lane, (3, 3), 3)
    # Canny 엣지 검출 알고리즘 적용
    edges = cv2.Canny(blurred, 50, 150)
    # Hough 변환으로 선분 감지
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=50)

   # 감지된 선분 처리
    if lines is not None:
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if abs(np.degrees(np.arctan(slope))) >= 25:
                # 기울기가 충분히 큰 경우에만 직선 고려
                if not merged_lines:
                    merged_lines.append([x1, y1, x2, y2])
                else:
                    merged = False
                    for merged_line in merged_lines:
                        x3, y3, x4, y4 = merged_line
                        distance = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                        angle1 = np.degrees(np.arctan(slope))
                        angle2 = np.degrees(np.arctan((y4 - y3) / (x4 - x3)))
                        angle_diff = abs(angle1 - angle2)
                        if distance < 10 and angle_diff < 10:
                            # 직선들이 서로 가까우면 병합
                            x1 = min(x1, x3)
                            y1 = min(y1, y3)
                            x2 = max(x2, x4)
                            y2 = max(y2, y4)
                            merged_lines.remove(merged_line)
                            merged_lines.append([x1, y1, x2, y2])
                            merged = True
                            break
                    if not merged:
                        merged_lines.append([x1, y1, x2, y2])

    # 프레임의 높이, 너비 추출
    frame_height, frame_width, _ = resized_frame.shape
    # 각 차선의 데이터를 저장할 리스트 초기화
    line_data = []
    for merged_lines in prev_frames_lines:
        for line in merged_lines:
            x1, y1, x2, y2 = line
            # 차선의 중심점 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # 차선의 각도 계산
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_data.append([center_x, center_y, angle])

    # 프레임 중앙을 기준으로 차선을 두 그룹으로 분류
    group_0_lines = [line for line in line_data if line[0] < frame_width / 2] 
    group_1_lines = [line for line in line_data if line[0] >= frame_width / 2]

    # 각 그룹의 평균 각도 계산
    group_angles = [np.mean([line[2] for line in group_0_lines if len(group_0_lines) > 0]), 
                    np.mean([line[2] for line in group_1_lines if len(group_1_lines) > 0])]

    # 각 그룹의 선분을 시각화
    for i in range(2):
        if i == 0 and group_0_lines:
            center_x, center_y, _ = np.mean(group_0_lines, axis=0)
        elif i == 1 and group_1_lines:
            center_x, center_y, _ = np.mean(group_1_lines, axis=0)
        else:
            continue
        angle = group_angles[i]
        angle_rad = np.radians(angle)
        line_length = int(1 * frame_width)
        x1 = int(center_x - 0.5 * line_length * np.cos(angle_rad))
        y1 = int(center_y - 0.5 * line_length * np.sin(angle_rad))
        x2 = int(center_x + 0.5 * line_length * np.cos(angle_rad))
        y2 = int(center_y + 0.5 * line_length * np.sin(angle_rad))
        cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(roi, "Center", (int(frame_width/2), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(roi, "Distance: {:.2f}".format(frame_width/2 - center_x), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
        # 중심점과의 거리를 기준으로 차량의 방향 제어
        distance = frame_width / 2 - center_x
        control_direction(distance)

    # 감지된 선분의 수와 현재 속도를 파란색으로 화면에 표시
    detected_lines_count = len(lines) if lines is not None else 0
    cv2.putText(roi, "Speed: {} RPM".format(current_speed), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(roi, "Lines Detected: {}".format(detected_lines_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 처리된 프레임 화면에 표시
    cv2.imshow("Lane Detection", resized_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 연결 해제 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
