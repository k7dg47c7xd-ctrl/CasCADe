import React, { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";
import * as THREE from "three";
import AxisCorner from "./AxisCorner";

const MODEL_TARGET_SIZE = 2.6;

function colorFromRgba(rgba) {
  if (Array.isArray(rgba) && rgba.length >= 3) {
    const to01 = (v) => (typeof v === "number" && v > 1 ? v / 255 : v);
    return new THREE.Color(to01(rgba[0]), to01(rgba[1]), to01(rgba[2]));
  }
  return new THREE.Color("#9ec0e0");
}

function normalizeGeometryForViewer(sourceGeometry, targetSize = MODEL_TARGET_SIZE) {
  const geometry = sourceGeometry.clone();
  geometry.computeBoundingBox();

  const box = geometry.boundingBox;
  if (!box || box.isEmpty()) {
    geometry.computeVertexNormals();
    return { geometry, scale: 1 };
  }

  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  geometry.translate(-center.x, -center.y, -center.z);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = Number.isFinite(maxDim) && maxDim > 0 ? targetSize / maxDim : 1;

  return { geometry, scale };
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

function StlMesh({ url, color }) {
  const sourceGeometry = useLoader(STLLoader, url);

  const { geometry, scale } = useMemo(
    () => normalizeGeometryForViewer(sourceGeometry, MODEL_TARGET_SIZE),
    [sourceGeometry]
  );

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: colorFromRgba(color),
        metalness: 0.2,
        roughness: 0.6,
      }),
    [color]
  );

  return <mesh geometry={geometry} material={material} scale={scale} />;
}

export default function StlViewer({ url, color }) {
  const [axisQuaternion, setAxisQuaternion] = useState(() => new THREE.Quaternion());

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [5, 4, 5], fov: 45, near: 0.01, far: 1000 }}>
        <ambientLight intensity={1.25} />
        <directionalLight position={[4, 5, 6]} intensity={1.6} />
        <directionalLight position={[-4, -3, -5]} intensity={0.45} />
        <Suspense fallback={null}>
          <StlMesh url={url} color={color} />
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
    </div>
  );
}
