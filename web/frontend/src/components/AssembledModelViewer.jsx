import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";
import * as THREE from "three";
import occtimportjs from "occt-import-js";
import occtWasmUrl from "occt-import-js/dist/occt-import-js.wasm?url";
import AxisCorner from "./AxisCorner";

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

function StepPart({ partKey, url, color, onStateChange }) {
  const [group, setGroup] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStep() {
      setGroup(null);
      onStateChange(partKey, true, "");

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

      if (!cancelled) {
        setGroup(root);
        onStateChange(partKey, false, "");
      }
    }

    loadStep().catch((err) => {
      console.error(err);
      if (!cancelled) {
        onStateChange(partKey, false, err.message || "STEP loading failed");
      }
    });

    return () => {
      cancelled = true;
    };
  }, [partKey, url, color, onStateChange]);

  if (!group) return null;

  return <primitive object={group} />;
}

function StlPart({ partKey, url, color, onStateChange }) {
  const sourceGeometry = useLoader(STLLoader, url);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: colorFromRgba(color),
        metalness: 0.2,
        roughness: 0.6,
      }),
    [color]
  );

  useEffect(() => {
    onStateChange(partKey, false, "");
  }, [partKey, onStateChange, sourceGeometry]);

  return <mesh geometry={sourceGeometry} material={material} />;
}

function ModelPart({ model, onStateChange }) {
  if (model.type === "stl") {
    return (
      <StlPart
        partKey={model.key}
        url={model.url}
        color={model.color}
        onStateChange={onStateChange}
      />
    );
  }

  return (
    <StepPart
      partKey={model.key}
      url={model.url}
      color={model.color}
      onStateChange={onStateChange}
    />
  );
}

function SceneFitter({ groupRef, parts, loaded }) {
  useEffect(() => {
    if (!groupRef.current || !parts.length || loaded !== parts.length) return;

    const frameId = requestAnimationFrame(() => {
      if (!groupRef.current) return;
      groupRef.current.scale.setScalar(1);
      groupRef.current.position.set(0, 0, 0);
      fitObjectToViewer(groupRef.current, MODEL_TARGET_SIZE);
    });

    return () => cancelAnimationFrame(frameId);
  }, [groupRef, parts, loaded]);

  return null;
}

export default function AssembledModelViewer({ models = [] }) {
  const [axisQuaternion, setAxisQuaternion] = useState(() => new THREE.Quaternion());
  const [state, setState] = useState({ loaded: new Set(), errors: [] });
  const groupRef = useRef(null);

  const notifyPartState = useCallback((partKey, loading, error) => {
    setState((prev) => {
      const loaded = new Set(prev.loaded);
      if (loading) {
        loaded.delete(partKey);
      } else {
        loaded.add(partKey);
      }

      return {
        loaded,
        errors: error ? [...prev.errors, error] : prev.errors,
      };
    });
  }, []);

  const allLoaded = state.loaded.size >= models.length;
  const hasError = state.errors.length > 0;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas camera={{ position: [5, 4, 5], fov: 45, near: 0.01, far: 1000 }}>
        <ambientLight intensity={1.25} />
        <directionalLight position={[4, 5, 6]} intensity={1.6} />
        <directionalLight position={[-4, -3, -5]} intensity={0.45} />

        <group ref={groupRef}>
          {models.map((model) => (
            <Suspense key={model.key} fallback={null}>
              <ModelPart model={model} onStateChange={notifyPartState} />
            </Suspense>
          ))}
        </group>

        <SceneFitter
          groupRef={groupRef}
          parts={models}
          loaded={state.loaded.size}
        />

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

      {models.length > 0 && !allLoaded && !hasError && (
        <div
          className="viewer-empty"
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        >
          <div className="viewer-empty-main">
            Loading assembled model parts...
          </div>
        </div>
      )}

      {hasError && (
        <div
          className="viewer-empty"
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        >
          <div className="viewer-empty-main">
            {state.errors[state.errors.length - 1]}
          </div>
        </div>
      )}
    </div>
  );
}
