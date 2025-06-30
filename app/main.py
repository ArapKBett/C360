from flask import Blueprint, render_template, request, jsonify
from .model import predict_url
from .network import scan_network
from .file_integrity import check_file_integrity
from .password import analyze_password
from .vulnerability import scan_vulnerabilities

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/url', methods=['GET', 'POST'])
def url():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        try:
            prediction, confidence = predict_url(url)
            return jsonify({
                'url': url,
                'is_malicious': bool(prediction),
                'confidence': float(confidence)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('url.html')

@main_bp.route('/network', methods=['GET', 'POST'])
def network():
    if request.method == 'POST':
        target = request.form.get('target')
        if not target:
            return jsonify({'error': 'No target provided'}), 400
        try:
            result = scan_network(target)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('network.html')

@main_bp.route('/file', methods=['GET', 'POST'])
def file():
    if request.method == 'POST':
        file_path = request.form.get('file_path')
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        try:
            result = check_file_integrity(file_path)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('file.html')

@main_bp.route('/password', methods=['GET', 'POST'])
def password():
    if request.method == 'POST':
        password = request.form.get('password')
        if not password:
            return jsonify({'error': 'No password provided'}), 400
        try:
            result = analyze_password(password)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('password.html')

@main_bp.route('/vulnerability', methods=['GET', 'POST'])
def vulnerability():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        try:
            result = scan_vulnerabilities(url)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('vulnerability.html')
