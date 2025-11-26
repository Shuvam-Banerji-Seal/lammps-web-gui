# LAMMPS Webview

A web-based viewer for visualizing LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) data files in 3D.

![LAMMPS Webview Screenshot](https://raw.githubusercontent.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/main/public/screenshot.png)

## Features

*   **3D Visualization:** Renders atom positions and bonds from LAMMPS `.data` files in a 3D canvas.
*   **File Upload:** Upload your own LAMMPS data files to visualize them.
*   **Paste Data:** Directly paste the content of a data file into the text area.
*   **Example Data:** Load a pre-configured example of a C60 buckyball.
*   **Appearance Customization:**
    *   Switch between dark and light themes.
    *   Adjust material styles for atoms and bonds (e.g., standard, toon, shiny).
    *   Control the scale of atom radii and bond sizes.

## Technologies Used

*   **Frontend:**
    *   [React](https://react.dev/)
    *   [TypeScript](https://www.typescriptlang.org/)
    *   [Vite](https://vitejs.dev/)
    *   [Three.js](https://threejs.org/) & [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) for 3D rendering.
    *   [Tailwind CSS](https://tailwindcss.com/) for styling.
*   **Backend:**
    *   [Python](https://www.python.org/)
    *   [Flask](https://flask.palletsprojects.com/) for serving the example data file.

## Setup and Installation

To run this project locally, you will need to set up both the frontend and backend services.

**1. Clone the repository:**

```bash
git clone https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer.git
cd lammps_data_web_viewer
```

**2. Set up the Frontend:**

Install the required Node.js packages using `npm`.

```bash
npm install
```

**3. Set up the Backend:**

Install the required Python packages using `pip`. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## How to Run

**1. Run the Backend Server:**

Start the Flask server to handle requests for the example data.

```bash
python server.py
```

The backend server will run on `http://127.0.0.1:5000`.

**2. Run the Frontend Application:**

In a new terminal, start the Vite development server.

```bash
npm run dev
```

The frontend application will be accessible at `http://localhost:5173`. Open this URL in your web browser.

## Project Structure

```
/
├─── public/            # Static assets
├─── src/               # Deprecated source folder
├─── components/        # React components for the 3D view
│   ├─── AtomMesh.tsx   # Renders individual atoms
│   ├─── BondMesh.tsx   # Renders bonds between atoms
│   └─── MoleculeCanvas.tsx # Main 3D canvas and rendering logic
├─── services/
│   └─── parser.ts      # Core logic for parsing LAMMPS data files
├─── .github/workflows/ # CI/CD configuration
│   └─── deploy.yml     # GitHub Actions workflow for deployment
├─── App.tsx            # Main application component and UI layout
├─── server.py          # Flask backend for serving example data
├─── package.json       # Frontend dependencies and scripts
└─── requirements.txt   # Backend Python dependencies
```

## Deployment

The project is configured for continuous deployment to GitHub Pages using GitHub Actions. The workflow in `.github/workflows/deploy.yml` automates the following steps:
1.  Sets up Node.js.
2.  Installs dependencies and builds the frontend application.
3.  Deploys the built static files to the `gh-pages` branch.

**Note:** The live deployment only includes the static frontend. The "Load C60 Example" feature, which relies on the Python backend, will not work on the deployed GitHub Pages site. To use this feature, you must run the project locally with both frontend and backend servers active.
