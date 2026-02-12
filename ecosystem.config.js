module.exports = {
    apps: [
        {
            name: "data-extraction-app",
            script: "backend/production_server.py",
            interpreter: "C:\\Users\\INTERN\\server\\Insurance_pdf_extractor\\venv\\Scripts\\python.exe",
            env: {
                PORT: 5000,
                FLASK_ENV: "production",
                PYTHONUTF8: "1"
            }
        }
    ]
};
