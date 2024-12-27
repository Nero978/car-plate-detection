from flask import Flask, jsonify, request, send_from_directory
import car_plate as cp
import uuid
import os

from flask_cors import CORS

# 创建一个 Flask 应用
app = Flask(__name__)

# 允许所有来源的跨域请求
CORS(app)

# API 路由
@app.route('/api/get-plate', methods=['POST'])
def get_data():
    if 'file' not in request.files:
        return jsonify({
            'code': 400,
            'msg': 'No file part',
            'data': None
        }), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'code': 400,
            'msg': 'No selected file',
            'data': None
        }), 400
    file_type = file.filename.split('.')[-1]
    file.filename = str(uuid.uuid4()) + '.' + file_type
    os.makedirs('media', exist_ok=True)
    file_path = 'media/' + file.filename
    file.save(file_path)
    data = cp.get_plate(file_path)
    if data is None:
        return jsonify({
            'code': 400,
            'msg': 'No plate detected',
            'data': None
        })
    data['video'] = file_path
    print(data)
    return jsonify({
        'code': 200,
        'msg': 'Success',
        'data': data
        })


# 静态文件路由
@app.route('/media/<filename>')
def uploaded_file(filename):
    return send_from_directory('../media', filename)

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True, port=3000)
