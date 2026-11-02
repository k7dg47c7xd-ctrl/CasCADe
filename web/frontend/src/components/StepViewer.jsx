import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import AxisCorner from "./AxisCorner";
import * as THREE from "three";
import occtimportjs from "occt-import-js";
import occtWasmUrl from "occt-import-js/dist/occt-import-js.wasm?url";

const MODEL_TARGET_SIZE = 2.6;

function colorFromRgba(rgba) {
  if (Array.isArray(rgba) && rgba.length >= 3) {
    const to01 = (v) => (typeof v === "number" && v > 1 ? v / 255 : v);
    return new THREE.Color(to01(rgba[0]), to01(rgba[1]), to01(rgba[2]));
  }
  return new THREE.Color(0.72, 0.78, 0.84);
}

function fitObjectToViewer(object, targetSize = MODEL_TARGET_SIZE) {
  object.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;

  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(maxDim) || maxDim <= 0) return;

  const scale = targetSize / maxDim;

  // In Three.js, position is applied after scale, so multiply by scale here,
  // so the model's initial center lands near the world origin and its size fits the viewport.
  object.scale.setScalar(scale);
  object.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}

function CameraAxisSync({ onQuaternionChange }) {
  const { camera } = useThree();
  const lastQuaternionRef = useRef(new THREE.Quaternion());

  useFrame(() => {
    if (!lastQuaternionRef.current.equals(camera.quaternion)) {
      lastQuaternionRef.current.copy(camera.quaternion);
      onQuaternionChange(camera.quaternion.clone());
    }
  });

  return null;
}

function StepModel({ url, color, onLoaded, onError }) {
  const [group, setGroup] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStep() {
      setGroup(null);
      onError("");

      const occt = await occtimportjs({
        locateFile: (fileName) => {
          if (String(fileName).endsWith(".wasm")) {
            return occtWasmUrl;
          }
          return fileName;
        },
      });

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`STEP file request failed: ${response.status}`);
      }

      const buffer = await response.arrayBuffer();
      const result = occt.ReadStepFile(new Uint8Array(buffer), {
        linearUnit: "millimeter",
        linearDeflectionType: "bounding_box_ratio",
        linearDeflection: 0.001,
        angularDeflection: 0.5,
      });

      if (!result || !Array.isArray(result.meshes)) {
        throw new Error("STEP file parse failed: no mesh data was returned.");
      }

      const root = new THREE.Group();

      for (const resultMesh of result.meshes) {
        const geometry = new THREE.BufferGeometry();
        const position = resultMesh?.attributes?.position?.array;
        if (!position) continue;

        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(position, 3)
        );

        const normal = resultMesh?.attributes?.normal?.array;
        if (normal) {
          geometry.setAttribute(
            "normal",
            new THREE.Float32BufferAttribute(normal, 3)
          );
        } else {
          geometry.computeVertexNormals();
        }

        const index = resultMesh?.index?.array;
        if (index) {
          geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(index), 1));
        }

        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        const material = new THREE.MeshStandardMaterial({
          color: colorFromRgba(color),
          metalness: 0.1,
          roughness: 0.55,
          side: THREE.DoubleSide,
        });

        root.add(new THREE.Mesh(geometry, material));
      }

      if (!root.children.length) {
        throw new Error("STEP file parsed successfully, but no displayable mesh was generated.");
      }

      fitObjectToViewer(root, MODEL_TARGET_SIZE);

      if (!cancelled) {
        setGroup(root);
        onLoaded();
      }
    }

    loadStep().catch((err) => {
      console.error(err);
      if (!cancelled) onError(err.message || "STEP loading failed");
    });

    return () => {
      cancelled = true;
    };
  }, [url, color, onLoaded, onError]);

  if (!group) return null;

  return <primitive object={group} />;
}

function StepStatus({ url, status, error }) {
  const fileName = useMemo(
    () => decodeURIComponent(url.split("/").pop() || "STEP file"),
    [url]
  );

  if (status === "loaded") return null;

  return (
    <div
      className="viewer-empty"
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    >
      <div className="viewer-empty-main">
        {error ? `STEP loading failed: ${error}` : `Loading STEP model: ${fileName}`}
      </div>
    </div>
  );
}

export default function StepViewer({ url, color }) {
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [axisQuaternion, setAxisQuaternion] = useState(() => new THREE.Quaternion());

  const handleLoaded = useMemo(() => () => setStatus("loaded"), []);
  const handleError = useMemo(
    () => (message) => {
      setError(message || "");
      setStatus(message ? "error" : "loading");
    },
    []
  );

  useEffect(() => {
    setStatus("loading");
    setError("");
  }, [url]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [5, 4, 5], fov: 45, near: 0.01, far: 1000 }}>
        <ambientLight intensity={1.25} />
        <directionalLight position={[4, 5, 6]} intensity={1.6} />
        <directionalLight position={[-4, -3, -5]} intensity={0.45} />
        <Suspense fallback={null}>
          <StepModel url={url} color={color} onLoaded={handleLoaded} onError={handleError} />
        </Suspense>
        <OrbitControls
          enablePan
          enableZoom
          enableRotate
          target={[0, 0, 0]}
          makeDefault
        />
        <CameraAxisSync onQuaternionChange={setAxisQuaternion} />
      </Canvas>
      <AxisCorner cameraQuaternion={axisQuaternion} />
      <StepStatus url={url} status={status} error={error} />
    </div>
  );
}
