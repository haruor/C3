function createBuildings() {
    // Solid, light gray material for clear geometric visibility
    const material = new THREE.MeshStandardMaterial({ 
        color: 0xcccccc,
        roughness: 0.6,
        metalness: 0.1,
    });

    const width = 15;   // ビルの幅 (X方向)
    const depth = 15;   // ビルの奥行き (Z方向)
    const gap = 5;      // ビル間の隙間 (道路幅)
    const pitch = width + gap; // グリッドの間隔 (20m)

    // 全てのビルで共通のベースジオメトリを使用（スケールでサイズ変更）
    const baseGeometry = new THREE.BoxGeometry(1, 1, 1);

    // 奥方向(Z軸)のループ: Z=0〜180付近まで
    for (let z = depth / 2; z <= 180; z += pitch) {
        
        // そのZ位置における中央の空洞幅（半分）を計算
        let halfGap;
        if (z < 80) {
            // Zone A (漏斗部): Z=0で50m、Z=80で10mへ線形に狭まる
            halfGap = 50 - 40 * (z / 80);
        } else {
            // Zone B (路地部): 常に幅10m (左右合わせて20mのキャニオン)
            halfGap = 10;
        }

        // 左側 (Left Side) のビル配置
        // 最初のビルは中心から -halfGap - width/2 の位置からスタート
        let currentX_left = -halfGap - width / 2;
        while (currentX_left - width / 2 >= -75) {
            const height = 100 * (0.8 + Math.random() * 0.4); // 80m 〜 120m のランダムな高さ

            const mesh = new THREE.Mesh(baseGeometry, material);
            mesh.scale.set(width, height, depth);
            // Three.jsはジオメトリの中心が原点なので、Y軸(高さ)は height/2 に配置して地面に接置させる
            mesh.position.set(currentX_left, height / 2, z);
            
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
            
            // 次のビルへ（X軸マイナス方向へ）
            currentX_left -= pitch;
        }

        // 右側 (Right Side) のビル配置
        let currentX_right = halfGap + width / 2;
        while (currentX_right + width / 2 <= 75) {
            const height = 100 * (0.8 + Math.random() * 0.4); // 80m 〜 120m のランダムな高さ

            const mesh = new THREE.Mesh(baseGeometry, material);
            mesh.scale.set(width, height, depth);
            mesh.position.set(currentX_right, height / 2, z);
            
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
            
            // 次のビルへ（X軸プラス方向へ）
            currentX_right += pitch;
        }
    }
}
