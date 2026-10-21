function createBuildings() {
    // Solid, light gray material for clear geometric visibility
    const material = new THREE.MeshStandardMaterial({
        color: 0xcccccc,
        roughness: 0.6,
        metalness: 0.1,
    });

    // Shape for Left Building footprint
    // Coordinates represent (X, Z) in the 2D plane before extrusion
    const leftShape = new THREE.Shape();
    leftShape.moveTo(-75, 0);
    leftShape.lineTo(-75, -180);  // Alley end outer
    leftShape.lineTo(-10, -180);  // Alley end inner
    leftShape.lineTo(-10, -80);   // Funnel exit inner
    leftShape.lineTo(-50, 0);     // Entrance inner
    leftShape.lineTo(-75, 0);     // Back to start

    // Shape for Right Building footprint
    const rightShape = new THREE.Shape();
    rightShape.moveTo(75, 0);
    rightShape.lineTo(50, 0);
    rightShape.lineTo(10, -80);
    rightShape.lineTo(10, -180);
    rightShape.lineTo(75, -180);
    rightShape.lineTo(75, 0);

    const extrudeSettings = {
        depth: 100, // This will become the building height after rotation
        bevelEnabled: false
    };

    const leftGeometry = new THREE.ExtrudeGeometry(leftShape, extrudeSettings);
    const rightGeometry = new THREE.ExtrudeGeometry(rightShape, extrudeSettings);

    const leftMesh = new THREE.Mesh(leftGeometry, material);
    const rightMesh = new THREE.Mesh(rightGeometry, material);

    // Rotate shapes by -90 degrees around X-axis
    // Extruded depth becomes positive Y (height)
    // 2D Shape Y becomes positive Z (depth)
    leftMesh.rotation.x = -Math.PI / 2;
    rightMesh.rotation.x = -Math.PI / 2;

    // Enable casting and receiving shadows for depth perception
    leftMesh.castShadow = true;
    leftMesh.receiveShadow = true;
    rightMesh.castShadow = true;
    rightMesh.receiveShadow = true;

    scene.add(leftMesh);
    scene.add(rightMesh);
}
