from flask import Flask, request, jsonify
import os

from sexiest_diffs import summary_from_files, persona_from_role

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

    all_summaries = {}

    for earlier_file, later_file in zip(earlier, later):
        earlier_file.save(os.path.join(UPLOAD_FOLDER, earlier_file.filename))
        later_file.save(os.path.join(UPLOAD_FOLDER, later_file.filename))

        # Handle Ashley's stuff
        persona = persona_from_role(role)
        role_summaries, prompt_tokens, response_tokens = summary_from_files(os.path.join(UPLOAD_FOLDER, earlier_file.filename), os.path.join(UPLOAD_FOLDER, later_file.filename), {role: persona})
        all_summaries.update({later_file.filename: role_summaries[role]})

    return jsonify({"summaries": all_summaries})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
