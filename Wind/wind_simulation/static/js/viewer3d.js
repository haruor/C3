class Viewer3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // シーン初期化
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000510); // 深い夜空のような黒青

        // カメラ初期化
        this.camera = new THREE.PerspectiveCamera(50, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
        this.camera.position.set(-80, 50, 120);

        // レンダラー初期化
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.container.appendChild(this.renderer.domElement);

        // OrbitControls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 10, 0);

        // 照明設定
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(50, 100, 50);
        this.scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x38bdf8, 2, 200);
        pointLight.position.set(0, 30, 20);
        this.scene.add(pointLight);

        // --- 靄（Fog/Haze）システム初期化 ---
        this.numStreamers = 3000; // 線の時の8000から減らし、一つ一つの粒を大きくする
        this.streamers = [];
        this.uArray = null;
        this.nx = 400;
        this.ny = 100;

        // ジオメトリとバッファの設定
        this.streamGeo = new THREE.BufferGeometry();
        this.positions = new Float32Array(this.numStreamers * 3); // 1頂点 x 3成分(x,y,z)
        this.colors = new Float32Array(this.numStreamers * 3);

        // パーティクルの初期配置
        for (let i = 0; i < this.numStreamers; i++) {
            this.respawnStreamer(i, true);
        }

        this.streamGeo.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        this.streamGeo.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

        // ソフトパーティクル用のテクスチャ生成
        const particleTexture = this.createParticleTexture();

        // マテリアル設定（四角い完全なソリッド粒子）
        const streamMat = new THREE.PointsMaterial({
            size: 1.5,
            vertexColors: true,
            transparent: false, // 不透明度0（＝透明度ゼロ、完全にソリッド）
            depthWrite: true
        });

        this.streamPoints = new THREE.Points(this.streamGeo, streamMat);
        this.scene.add(this.streamPoints);

        // --- 3Dパイプの材質 ---
        this.pipeMat = new THREE.MeshStandardMaterial({
            color: 0x64748b,
            roughness: 0.2,
            metalness: 0.8,
            side: THREE.DoubleSide
        });
        this.pipeMeshes = [];
        this.pipesConfig = [];

        window.addEventListener('resize', this.onResize.bind(this));

        // アニメーションループ開始
        this.animate = this.animate.bind(this);
        this.animate();
    }

    createParticleTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');

        // 中心が白く、外側に向かって透明になる円形グラデーション
        const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.2, 'rgba(255, 255, 255, 0.8)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.2)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 64, 64);

        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }

    respawnStreamer(i, fullRandom = false) {
        let x, y;
        const rand = Math.random();

        if (this.pipesConfig && this.pipesConfig.length > 0) {
            // ランダムに1つのパイプを選ぶ
            const pipe = this.pipesConfig[Math.floor(Math.random() * this.pipesConfig.length)];
            let pipeYBottom = this.pipeYTop + pipe.depth;

            if (rand < 0.3) {
                // パイプ内部に発生 (空気が揺れているのを見せるため)
                x = pipe.x + this.thickness + Math.random() * pipe.width;
                y = this.pipeYTop + Math.random() * pipe.depth;
            } else if (rand < 0.7) {
                // 筒のすぐ上流 (渦を形成する気流)
                x = pipe.x - 20 + Math.random() * 40; // 複数あるので上流範囲を少し狭める
                y = this.pipeYTop - 10 + Math.random() * 20;
            } else {
                // 空間全体
                x = fullRandom ? Math.random() * this.nx : 0.0;
                y = Math.random() * this.ny;
            }
        } else {
            x = fullRandom ? Math.random() * this.nx : 0.0;
            y = Math.random() * this.ny;
        }

        let z = (Math.random() - 0.5) * 80;
        this.streamers[i] = { x, y, z, life: Math.random() };

        const mapX = (vx) => vx - 200;
        const mapY = (vy) => 50 - vy;

        const idx = i * 3;
        const posX = mapX(x);
        const posY = mapY(y);

        // 頂点更新
        this.positions[idx] = posX;
        this.positions[idx + 1] = posY;
        this.positions[idx + 2] = z;
    }

    updatePipes(pipesConfig, pipeYTop, thickness) {
        // configが変更されていない場合は再描画をスキップ（毎フレームの負荷軽減）
        const newConfigStr = JSON.stringify(pipesConfig) + pipeYTop + thickness;
        if (this._lastPipesConfigStr === newConfigStr) return;
        this._lastPipesConfigStr = newConfigStr;

        this.pipesConfig = pipesConfig;
        this.pipeYTop = pipeYTop;
        this.thickness = thickness;

        // 古いパイプメッシュを削除
        if (this.pipeMeshes) {
            this.pipeMeshes.forEach(mesh => {
                this.scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                mesh.children.forEach(child => {
                    if (child.geometry) child.geometry.dispose();
                });
            });
        }
        this.pipeMeshes = [];

        const mapX = (x) => x - 200;
        const mapY = (y) => 50 - y;

        pipesConfig.forEach(pipe => {
            const width = pipe.width;
            const depth = pipe.depth;
            const pipeX = pipe.x;
            const pipeHeight = depth + thickness;

            // パイプの中心X座標（LBM座標系）
            const centerX = mapX(pipeX + thickness + width / 2);
            const topY = mapY(pipeYTop);

            // 円柱の半径（壁の厚みを考慮）
            const outerRadius = (width + thickness * 2) / 2;

            // THREE.CylinderGeometry(radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded)
            // openEnded=true にして中空（パイプ状）にする
            const cylGeo = new THREE.CylinderGeometry(outerRadius, outerRadius, pipeHeight, 32, 1, true);
            const cylinder = new THREE.Mesh(cylGeo, this.pipeMat);
            // CylinderGeometry はデフォルトでY軸方向に伸びるのでそのまま配置
            cylinder.position.set(centerX, topY - pipeHeight / 2, 0);

            this.scene.add(cylinder);
            this.pipeMeshes.push(cylinder);

            // パイプの底のフタ (CircleGeometry)
            const bottomGeo = new THREE.CircleGeometry(outerRadius, 32);
            const bottomCap = new THREE.Mesh(bottomGeo, this.pipeMat);
            bottomCap.rotation.x = Math.PI / 2; // 水平に倒す
            // 底の位置（Y軸方向の下端）
            bottomCap.position.set(centerX, topY - pipeHeight, 0);

            this.scene.add(bottomCap);
            this.pipeMeshes.push(bottomCap);
        });
    }

    updateFluid(uArray, nx, ny) {
        this.uArray = uArray;
        this.nx = nx;
        this.ny = ny;
    }

    onResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(this.animate);

        // --- 流線の物理アニメーション ---
        if (this.uArray && this.nx && this.ny) {
            const mapX = (vx) => vx - 200;
            const mapY = (vy) => 50 - vy;
            // 速度の視覚的な倍率（小さくすると移動は遅くなるが、軌跡はtailScaleで伸ばす）
            const speedScale = 40.0;
            const tailScale = 3.0;

            for (let i = 0; i < this.numStreamers; i++) {
                let p = this.streamers[i];

                // LBM配列上のインデックス
                let gridX = Math.floor(p.x);
                let gridY = Math.floor(p.y);

                let ux = 0.05; // デフォルトのベース風速
                let uy = 0.0;

                if (gridX >= 0 && gridX < this.nx && gridY >= 0 && gridY < this.ny) {
                    const idxX = gridY * this.nx + gridX;
                    const idxY = this.nx * this.ny + gridY * this.nx + gridX;
                    ux = this.uArray[idxX];
                    uy = this.uArray[idxY];
                }

                const dx = ux * speedScale;
                const dy = uy * speedScale;

                // 速度によるパーティクルのゆらぎ効果（もやもや感）を追加
                const jitterX = (Math.random() - 0.5) * 0.5;
                const jitterY = (Math.random() - 0.5) * 0.5;

                // update position
                p.x += dx + jitterX;
                p.y += dy + jitterY;
                // パーティクルの寿命減衰
                p.life -= 0.003;

                // 画面外や寿命が尽きたらリスポーン
                // LBMモデルは右端から流出するため x > nx でリセット
                if (p.x >= this.nx || p.x < 0 || p.y < 0 || p.y >= this.ny || p.life <= 0) {
                    this.respawnStreamer(i, false);
                    continue;
                }

                const bufIdx = i * 3;
                // current vertex
                this.positions[bufIdx] = mapX(p.x);
                this.positions[bufIdx + 1] = mapY(p.y);
                this.positions[bufIdx + 2] = p.z;

                // 色の計算 (風速の絶対値でヒートマップ色付け: 遅い=青, 中=緑/黄, 早い=赤)
                const speed = Math.sqrt(ux * ux + uy * uy);
                const normSpeed = Math.min(speed / 0.1, 1.0);

                let r, g, b;
                if (normSpeed < 0.33) {
                    // 青からシアンへ
                    let t = normSpeed / 0.33;
                    r = 0.0; g = t; b = 1.0;
                } else if (normSpeed < 0.66) {
                    // シアンから黄色へ
                    let t = (normSpeed - 0.33) / 0.33;
                    r = t; g = 1.0; b = 1.0 - t;
                } else {
                    // 黄色から赤へ
                    let t = (normSpeed - 0.66) / 0.34;
                    r = 1.0; g = 1.0 - t; b = 0.0;
                }

                // lifeに基づくフェードイン・フェードアウト効果
                let alpha = 1.0;
                if (p.life > 0.8) alpha = (1.0 - p.life) / 0.2;
                else if (p.life < 0.2) alpha = p.life / 0.2;

                this.colors[bufIdx] = r * alpha;
                this.colors[bufIdx + 1] = g * alpha;
                this.colors[bufIdx + 2] = b * alpha;
            }

            this.streamGeo.attributes.position.needsUpdate = true;
            this.streamGeo.attributes.color.needsUpdate = true;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
