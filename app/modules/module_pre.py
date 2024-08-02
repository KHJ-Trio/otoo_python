import re
from datetime import datetime

def clean_chat(data):
    # 날짜 인식
    date_pattern = re.compile(r'^-+ (\d{4}년 \d{1,2}월 \d{1,2}일) .+ -+$')
    # 카톡 메시지 인식
    message_pattern = re.compile(r'^\[(.+?)\] \[(오전|오후) (\d{1,2}:\d{2})\] (.+)$')
    # ㅋ 패턴 인식
    k_pattern = re.compile(r'ㅋ{2,}')
    # ? 세개 이상 인식
    question_pattern = re.compile(r'\?{3,}')
    # ! 세개 이상 인식
    exclamation_pattern = re.compile(r'\!{3,}')
    # URL 인식
    url_pattern = re.compile(r'http\S+')
    # 이모티콘 인식
    emoj_pattern = re.compile(r'이모티콘+')

    clean_lines = []
    current_date = ""
    current_speaker = ""
    current_time = ""
    current_message = []

    def add_message():
        if current_speaker and current_message:
            cleaned_message = ' '.join(current_message)
            # ㅋㅋ 2개까지만
            cleaned_message = k_pattern.sub('ㅋㅋ', cleaned_message)
            # ?? 2개까지만
            cleaned_message = question_pattern.sub('??', cleaned_message)
            # !! 2개까지만
            cleaned_message = exclamation_pattern.sub('!!', cleaned_message)
            # URL 메시지 변경
            cleaned_message = url_pattern.sub('정보 공유', cleaned_message)
            # 이모티콘 메시지 삭제
            if not emoj_pattern.search(cleaned_message):
                clean_lines.append(f'{current_date} {current_time} {current_speaker} {cleaned_message}')
            current_message.clear()

    for line in data.split('\n'):
        line = line.strip()
        date_match = date_pattern.match(line)
        if date_match:
            add_message()
            # 날짜 형식 '24/00/00'
            raw_date = date_match.group(1)
            formatted_date = datetime.strptime(raw_date, '%Y년 %m월 %d일').strftime('%y/%m/%d')
            current_date = formatted_date
            current_speaker = ""
            current_time = ""
            current_message = []
            continue
        message_match = message_pattern.match(line)
        if message_match:
            add_message()
            # 유저 이름
            current_speaker = f'[{message_match.group(1)}]'
            am_pm = 'AM' if message_match.group(2) == '오전' else 'PM'
            # 시간
            current_time = f'{am_pm} {message_match.group(3)}'
            # 메시지
            current_message = [message_match.group(4)]
        else:
            if current_speaker:
                current_message.append(line.strip())

    add_message()  # 메시지 추가
    cleaned_text = "\n".join(clean_lines)
    # print(cleaned_text)
    return cleaned_text
