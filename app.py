from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
import numpy as np
from joblib import load
import gensim.downloader as api
import random
from jinja2 import Environment, FileSystemLoader

app = Flask(__name__)

# Load the logistic regression model and category names
model = load('logistic_regression_model.pkl')
category_names = np.load('category_names.npy', allow_pickle=True)

# Load the Word2Vec model (use the same method you used in your script)
word2vec_googlenews = api.load('word2vec-google-news-300')

# Function to preprocess and transform input text
def preprocess_and_transform(title, description, word2vec_model):
    combined_text = title + " " + description
    words = combined_text.split()
    word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if word_vectors:
        doc_vector = np.mean(word_vectors, axis=0)
    else:
        doc_vector = np.zeros(word2vec_model.vector_size)
    return doc_vector.reshape(1, -1)

def get_latest_jobs(category, count=2):
    categories_dir = "templates/categories"
    category_path = os.path.join(categories_dir, category)
    job_files = sorted(
        [f for f in os.listdir(category_path) if f.endswith('.html')],
        key=lambda x: os.path.getmtime(os.path.join(category_path, x)),
        reverse=True
    )[:count]
    return [url_for('job', category=category, webindex=f.split('.')[0]) for f in job_files]

@app.route('/')
def index():
    latest_jobs = {
        'engineering': get_latest_jobs('engineering'),
        'finance': get_latest_jobs('finance'),
        'healthcare': get_latest_jobs('healthcare'),
        'sales': get_latest_jobs('sales')
    }
    return render_template('home.html', latest_jobs=latest_jobs)

@app.route('/employers', methods=['GET', 'POST'])
def employers():
    if request.method == 'POST':
        f_title = request.form['title']
        f_company = request.form['company']
        f_content = request.form['description']
        input_vector = preprocess_and_transform(f_title, f_content, word2vec_googlenews)
        predicted_category_idx = model.predict(input_vector)[0]
        predicted_category = category_names[predicted_category_idx]
        predicted_message = f"The suggested category of this job is {predicted_category}."
        return render_template('employers.html', predicted_message=predicted_message, title=f_title, company=f_company, description=f_content, ypred=predicted_category)
    else:
        return render_template('employers.html')

@app.route('/add_job', methods=['POST'])
def add_job():
    title = request.form['title']
    company = request.form['company']
    description = request.form['description']
    ypred = request.form['ypred']
    category_folder_map = {
        'Accounting & Finance': 'finance',
        'Engineering': 'engineering',
        'Healthcare & Nursing': 'healthcare',
        'Sales': 'sales'
    }
    category_folder = category_folder_map.get(ypred, ypred)
    create_job_html(title, company, description, category_folder)
    return """
    <script>
        alert("Job added successfully!");
        window.history.go(-1);
    </script>
    """

def create_job_html(title, company, description, ypred):
    categories_dir = "templates/categories"
    category_path = os.path.join(categories_dir, ypred)
    os.makedirs(category_path, exist_ok=True)
    random_number = random.randint(10000000, 99999999)
    file_path = os.path.join(category_path, f"{random_number}.html")
    template_loader = FileSystemLoader('templates/')  
    jinja2_env = Environment(loader=template_loader)
    template = jinja2_env.get_template('job_template.html')  # Ensure this is the correct template name
    html_content = template.render(title=title, company=company, description=description)
    with open(file_path, "w") as file:
        file.write(html_content)

@app.route('/categories')
def categories():
    return render_template('categories.html')

@app.route('/categories/<category>')
def category_page(category):
    job_urls = get_all_jobs_in_category(category)
    job_urls_json = json.dumps(job_urls)
    return render_template(f'categories/{category}.html', job_urls_json=job_urls_json)

@app.route('/categories/<category>/<webindex>')
def job(category, webindex):
    return render_template('categories/' + category + '/' + webindex + '.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', error=e), 500

def get_all_jobs_in_category(category):
    categories_dir = "templates/categories"
    job_urls = []
    category_path = os.path.join(categories_dir, category)

    if os.path.exists(category_path) and os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.endswith('.html'):
                webindex = filename.split('.')[0]
                job_url = f"/categories/{category}/{webindex}"
                job_urls.append(job_url)
    return job_urls

if __name__ == '__main__':
    app.run(debug=True)