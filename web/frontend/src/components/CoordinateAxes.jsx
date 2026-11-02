import React from "react";
import { Billboard, Line, Text } from "@react-three/drei";

function AxisLabel({ position, color, children, fontSize }) {
  return (
    <Billboard position={position} follow lockX={false} lockY={false} lockZ={false}>
      <Text
        fontSize={fontSize}
        color={color}
        anchorX="center"
        anchorY="middle"
        outlineWidth={fontSize * 0.08}
        outlineColor="#ffffff"
      >
        {children}
      </Text>
    </Billboard>
  );
}

export default function CoordinateAxes({ size = 2.4 }) {
  const axisLength = size;
  const negativeLength = size * 0.18;
  const coneRadius = size * 0.035;
  const coneHeight = size * 0.13;
  const labelOffset = size * 0.18;
  const labelSize = Math.max(size * 0.09, 0.12);

  return (
    <group>
      {/* Reference grid: on the XY plane, with Z as the vertical axis */}
      <gridHelper
        args={[size * 2.2, 10, "#d1d5db", "#edf2f7"]}
        rotation={[Math.PI / 2, 0, 0]}
      />

      {/* X axis: red */}
      <Line
        points={[[-negativeLength, 0, 0], [axisLength, 0, 0]]}
        color="#e74c3c"
        lineWidth={3}
      />
      <mesh position={[axisLength, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[coneRadius, coneHeight, 20]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>
      <AxisLabel
        position={[axisLength + labelOffset, 0, 0]}
        color="#e74c3c"
        fontSize={labelSize}
      >
        X
      </AxisLabel>

      {/* Y axis: green */}
      <Line
        points={[[0, -negativeLength, 0], [0, axisLength, 0]]}
        color="#2f9e44"
        lineWidth={3}
      />
      <mesh position={[0, axisLength, 0]}>
        <coneGeometry args={[coneRadius, coneHeight, 20]} />
        <meshStandardMaterial color="#2f9e44" />
      </mesh>
      <AxisLabel
        position={[0, axisLength + labelOffset, 0]}
        color="#2f9e44"
        fontSize={labelSize}
      >
        Y
      </AxisLabel>

      {/* Z axis: blue */}
      <Line
        points={[[0, 0, -negativeLength], [0, 0, axisLength]]}
        color="#2b6cb0"
        lineWidth={3}
      />
      <mesh position={[0, 0, axisLength]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[coneRadius, coneHeight, 20]} />
        <meshStandardMaterial color="#2b6cb0" />
      </mesh>
      <AxisLabel
        position={[0, 0, axisLength + labelOffset]}
        color="#2b6cb0"
        fontSize={labelSize}
      >
        Z
      </AxisLabel>
    </group>
  );
}
