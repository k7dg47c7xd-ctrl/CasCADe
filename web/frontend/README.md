# CasCADe: Conversational CAD Modeling Frontend

## 1. Project Overview

This directory contains the frontend implementation of CasCADe, a conversational Computer-Aided Design (CAD) modeling application developed using React and Vite.

The frontend provides an interactive interface that enables users to describe CAD modeling requirements in natural language and visualize the generated three-dimensional CAD models.

## 2. Main Features

The frontend provides the following functionalities:

- **Natural Language Interaction:** A chat-based interface for specifying CAD modeling requirements.
- **3D Model Visualization:** Interactive rendering of CAD models using Three.js and React Three Fiber.
- **CAD File Support:** Visualization of STL and STEP models.
- **Assembly Visualization:** Interactive visualization of multi-component CAD assemblies.
- **Model Construction Animation:** Animated visualization of CAD model construction processes.
- **Coordinate Visualization:** Coordinate axes and orientation indicators for improved spatial understanding.
- **Backend Communication:** API integration for conversational modeling and CAD generation.

## 3. Technologies

The frontend is developed using:

- React
- Vite
- Three.js
- @react-three/fiber
- @react-three/drei
- occt-import-js
- JavaScript

## 4. Installation

The `node_modules` directory is not included in this repository because it contains third-party dependencies that can be automatically reinstalled.

To install and run the frontend, follow these steps.

### Step 1: Install Node.js

Install Node.js and npm from:

https://nodejs.org/

### Step 2: Navigate to the Frontend Directory

Open a terminal and navigate to the `cad-chat-frontend` directory.

### Step 3: Install Dependencies

Run:

```bash
npm install
```

Alternatively, if `package-lock.json` is available, use:

```bash
npm ci
```

This will automatically install the required dependencies and recreate the `node_modules` directory.

### Step 4: Start the Development Server

Run:

```bash
npm run dev
```

The frontend will normally be accessible at:

http://localhost:5173

## 5. Backend Connection

The frontend communicates with the CasCADe backend through API endpoints defined in `src/services/api.js`.

The main endpoints include:

- `/api/chat`: Conversational CAD interaction
- `/api/generate`: CAD model generation
- `/api/select-json`: CAD JSON selection
- `/api/list-files`: Generated file retrieval

The development configuration uses a backend service running at:

http://localhost:8002

The Vite proxy configuration is defined in `vite.config.js`.

**Note:** A running backend service is required for full conversational CAD generation functionality. The frontend alone does not implement the backend CAD generation or language model services.

## 6. Production Build

To build the frontend application, run:

```bash
npm run build
```

The build output is generated in the `dist` directory.

## 7. Dependency Availability

The `node_modules` directory has been intentionally excluded from this repository to reduce repository size and avoid distributing third-party dependencies.

All required frontend dependencies are declared in `package.json`, and their resolved versions are recorded in `package-lock.json` when provided.

Users can restore the dependencies by running:

```bash
npm install
```

Therefore, excluding `node_modules` does not affect the availability of the frontend source code or its ability to be installed and built.

## 8. Reproducibility

The repository provides the frontend source code and configuration files necessary to install, build, and run the user interface.

To reproduce the complete conversational CAD modeling workflow, the corresponding backend services must also be available and configured appropriately.