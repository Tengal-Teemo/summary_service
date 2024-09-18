from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def main():
    return app.send_static_file('demo.html')

@app.route('/summarize', methods=['POST'])
def handle_upload():  # put application's code here
    if 'earlier' not in request.files:
        return 'No \"earlier\" File Part', 400

    if 'later' not in request.files:
        return 'No \"later\" File Part', 400

    if 'role' not in request.form:
        return 'No role provided', 400

    earlier = request.files.getlist('earlier')
    later = request.files.getlist('later')
    role = request.form['role']

    if len(earlier) != len(later):
        return 'Mismatched numbers of earlier and later files', 400

    for earlier_file, later_file in zip(earlier, later):
        earlier_file.save(os.path.join(UPLOAD_FOLDER, earlier_file.filename))
        later_file.save(os.path.join(UPLOAD_FOLDER, later_file.filename))

    # Handle Ashley's stuff
    # generate_summary_from_file_diffs_and_role(file_paths, role)

    fake_summaries = ["Lorem", "Ipsum"]
    return jsonify({"summaries": fake_summaries})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
