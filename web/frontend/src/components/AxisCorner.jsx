import React, { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Text } from "@react-three/drei";
import * as THREE from "three";

function toQuaternion(cameraQuaternion) {
  if (!cameraQuaternion) {
    return new THREE.Quaternion();
  }

  if (cameraQuaternion instanceof THREE.Quaternion) {
    return cameraQuaternion.clone();
  }

  return new THREE.Quaternion(
    cameraQuaternion.x ?? cameraQuaternion._x ?? 0,
    cameraQuaternion.y ?? cameraQuaternion._y ?? 0,
    cameraQuaternion.z ?? cameraQuaternion._z ?? 0,
    cameraQuaternion.w ?? cameraQuaternion._w ?? 1
  );
}

function AxisArrow({
  direction = [1, 0, 0],
  color = "#d84c4c",
  label = "X",
  labelPosition = [1.9, 0, 0],
}) {
  const dir = new THREE.Vector3(...direction).normalize();

  // This controls clarity: thicker lines and larger arrows
  const shaftLength = 1.22;
  const shaftRadius = 0.075;
  const headLength = 0.38;
  const headRadius = 0.18;

  const shaftPosition = dir.clone().multiplyScalar(shaftLength / 2);
  const headPosition = dir
    .clone()
    .multiplyScalar(shaftLength + headLength / 2);

  const quaternion = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir
  );

  return (
    <group>
      <mesh position={shaftPosition.toArray()} quaternion={quaternion}>
        <cylinderGeometry args={[shaftRadius, shaftRadius, shaftLength, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.08}
          roughness={0.32}
        />
      </mesh>

      <mesh position={headPosition.toArray()} quaternion={quaternion}>
        <coneGeometry args={[headRadius, headLength, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.08}
          roughness={0.28}
        />
      </mesh>

      <Text
        position={labelPosition}
        fontSize={0.4}
        color={color}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.045}
        outlineColor="#ffffff"
      >
        {label}
      </Text>
    </group>
  );
}

function NegativeAxisLine({ direction = [-1, 0, 0] }) {
  const dir = new THREE.Vector3(...direction).normalize();

  const length = 0.72;
  const radius = 0.04;
  const position = dir.clone().multiplyScalar(length / 2);

  const quaternion = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir
  );

  return (
    <mesh position={position.toArray()} quaternion={quaternion}>
      <cylinderGeometry args={[radius, radius, length, 18]} />
      <meshStandardMaterial
        color="#8c96a3"
        transparent
        opacity={0.95}
        metalness={0.02}
        roughness={0.45}
      />
    </mesh>
  );
}

function OriginBall() {
  return (
    <mesh>
      <sphereGeometry args={[0.12, 32, 32]} />
      <meshStandardMaterial
        color="#f6f8fa"
        metalness={0.02}
        roughness={0.35}
      />
    </mesh>
  );
}

function AxisGizmoScene({ cameraQuaternion }) {
  const groupRef = useRef();

  const axisQuaternion = useMemo(() => {
    const q = toQuaternion(cameraQuaternion);

    // Key point: use the inverse of the main camera quaternion so the top-right axes stay in sync with the model view.
    return q.invert();
  }, [cameraQuaternion]);

  useEffect(() => {
    if (!groupRef.current) return;
    groupRef.current.quaternion.copy(axisQuaternion);
  }, [axisQuaternion]);

  return (
    <>
      <ambientLight intensity={1.25} />
      <directionalLight position={[4, 5, 6]} intensity={1.35} />
      <directionalLight position={[-4, -3, 3]} intensity={0.65} />

      <group ref={groupRef}>
        <OriginBall />

        <AxisArrow
          direction={[1, 0, 0]}
          color="#d84c4c"
          label="X"
          labelPosition={[2.08, 0, 0]}
        />

        <AxisArrow
          direction={[0, 1, 0]}
          color="#3aa657"
          label="Y"
          labelPosition={[0, 2.08, 0]}
        />

        <AxisArrow
          direction={[0, 0, 1]}
          color="#3d6fd6"
          label="Z"
          labelPosition={[0, 0, 2.08]}
        />

        <NegativeAxisLine direction={[-1, 0, 0]} />
        <NegativeAxisLine direction={[0, -1, 0]} />
        <NegativeAxisLine direction={[0, 0, -1]} />
      </group>
    </>
  );
}

export default function AxisCorner({ cameraQuaternion }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        width: 138,
        height: 138,
        background: "rgba(255, 255, 255, 0.97)",
        border: "1px solid #c8d0da",
        borderRadius: 16,
        boxShadow: "0 6px 18px rgba(0, 0, 0, 0.13)",
        overflow: "hidden",
        zIndex: 20,
        pointerEvents: "none",
      }}
    >
      <Canvas
        orthographic
        camera={{
          position: [0, 0, 6],
          zoom: 31,
          near: 0.1,
          far: 100,
        }}
        gl={{
          alpha: true,
          antialias: true,
        }}
      >
        <color attach="background" args={["#ffffff"]} />
        <AxisGizmoScene cameraQuaternion={cameraQuaternion} />
      </Canvas>
    </div>
  );
}