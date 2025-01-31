﻿# Movies Recommendation System

![Preview of Movies Recommendation System](/preview.png)

A cutting-edge web application that recommends movies to users based on their preferences. This project combines the power of Flask and Python for the backend with a modern React + TypeScript frontend to deliver a seamless and responsive user experience.

## Features
- **Personalized Recommendations**: Suggests movies tailored to user preferences.
- **External API Integration**: Leverages the robust The Movie Database (TMDB) API for movie data.
- **User-Friendly Interface**: Intuitive design for easy navigation and usability.
- **Responsive Design**: Optimized for devices of all screen sizes.

## Technologies Used
- **Backend**: Flask (Python)
- **Frontend**: React + TypeScript
- **API**: The Movie Database (TMDB) API

## Installation

### Prerequisites
Ensure the following tools are installed on your system:
- Python 3.8+
- pip (Python package manager)
- Node.js and npm (for the frontend)
- Git

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/mohamad27911/MovieRecommendation.git
   cd MoviesRecommendation/flask-server
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   flask run
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm run dev
   ```

### Access the Application
- Open your browser and navigate to:
  - Backend: `http://127.0.0.1:5000/`
  - Frontend: `http://127.0.0.1:5173/` (default Vite port)

## Folder Structure
```
MoviesRecommendation/
├── flask-server/          # Flask application files
├── frontend/              # React + TypeScript frontend
│   ├── src/               # Frontend source code
│   ├── assets/            # Images and static files
│   └── ...               
├── requirements.txt       # Backend dependencies
└── README.md              # Project documentation
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the [MIT License](LICENSE).
