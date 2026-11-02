import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";

function AnimatedBoxBuild({
  width = 1.0169484941854339,
  depth = 1.4999994185728851,
  halfHeight = 0.5498385585587136,
  mode = "full", // "full" | "positive" | "negative"
  color = "#58a6ff",
  offset = [0, 0, 0],
}) {
  const topRef = useRef();
  const bottomRef = useRef();
  const finalRef = useRef();
  const groupRef = useRef();

  const rectPoints = useMemo(() => {
    const x = width / 2;
    const y = depth / 2;

    return [
      [-x, -y, 0],
      [x, -y, 0],
      [x, y, 0],
      [-x, y, 0],
      [-x, -y, 0],
    ];
  }, [width, depth]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const time = Math.min(t, 8);

    const topProgress = THREE.MathUtils.clamp((time - 2) / 2, 0, 1);
    const bottomProgress = THREE.MathUtils.clamp((time - 4) / 2, 0, 1);
    const fuseProgress = THREE.MathUtils.clamp((time - 6) / 2, 0, 1);

    if (groupRef.current && t < 8) {
      groupRef.current.rotation.z = Math.sin(t * 0.25) * 0.08;
    }

    if (topRef.current) {
      if (mode === "positive" || mode === "full") {
        topRef.current.scale.z = Math.max(topProgress, 0.001);
        topRef.current.position.z = (halfHeight * topProgress) / 2;
        topRef.current.visible =
          mode === "full" ? time >= 2 && time < 6.8 : time >= 2;
      } else {
        topRef.current.visible = false;
      }
    }

    if (bottomRef.current) {
      if (mode === "negative") {
        bottomRef.current.scale.z = Math.max(topProgress, 0.001);
        bottomRef.current.position.z = -(halfHeight * topProgress) / 2;
        bottomRef.current.visible = time >= 2;
      } else if (mode === "full") {
        bottomRef.current.scale.z = Math.max(bottomProgress, 0.001);
        bottomRef.current.position.z = -(halfHeight * bottomProgress) / 2;
        bottomRef.current.visible = time >= 4 && time < 6.8;
      } else {
        bottomRef.current.visible = false;
      }
    }

    if (finalRef.current) {
      finalRef.current.visible = mode === "full" ? time >= 6 : false;
      finalRef.current.material.opacity = fuseProgress * 0.72;
    }
  });

  return (
    <group ref={groupRef} position={offset}>
      <Line points={rectPoints} color={color} lineWidth={3} />

      <mesh position={[0, 0, -0.002]}>
        <boxGeometry args={[width, depth, 0.004]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.22}
          depthWrite={false}
        />
      </mesh>

      <mesh ref={topRef}>
        <boxGeometry args={[width, depth, halfHeight]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.52}
          depthWrite={false}
        />
      </mesh>

      <mesh ref={bottomRef}>
        <boxGeometry args={[width, depth, halfHeight]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.52}
          depthWrite={false}
        />
      </mesh>

      <mesh ref={finalRef}>
        <boxGeometry args={[width, depth, halfHeight * 2]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.72}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

export default function CadBuildAnimation({ dimensions, models }) {
  const modelList =
    Array.isArray(models) && models.length > 0
      ? models.map((model) => ({
          id: model.id,
          name: model.name,
          dimensions: model.dimensions || {},
        }))
      : [
          {
            id: "single-model",
            name: "single-model",
            dimensions: dimensions || {},
          },
        ];

  return (
    <Canvas camera={{ position: [2.4, 2.4, 1.8], fov: 45 }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[3, 3, 4]} intensity={1.5} />

      {modelList.map((model, index) => {
        const d = model.dimensions || {};

        const width = d.width || 1.0169484941854339;
        const depth = d.depth || 1.4999994185728851;
        const halfHeight = d.halfHeight || 0.5498385585587136;
        const mode = d.mode || "full";
        const color = d.color || "#58a6ff";

        // When several models overlap, add a small offset so they do not coincide exactly and the layering remains visible
        const offsetAmount = index * 0.015;
        const offset = [offsetAmount, offsetAmount, 0];

        return (
          <AnimatedBoxBuild
            key={model.id || index}
            width={width}
            depth={depth}
            halfHeight={halfHeight}
            mode={mode}
            color={color}
            offset={offset}
          />
        );
      })}

      <OrbitControls enablePan enableZoom enableRotate />
    </Canvas>
  );
}