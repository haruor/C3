// 画像プレビューを表示する関数
function previewImage(event) {
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('preview-image');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
}

// ローディング画面を表示する関数
function showLoading() {
    // HTMLのdata属性から読み取るのをやめ、正しい絶対パスを直接書き込む
    const loadingStages = [
        {
            gif: '/static/images/rocket_liftoff.gif', // ★★★ パスを直接記述 ★★★
            text: 'リフトオフ！大気圏を突破中！',
            subtext: 'ぐんぐんスピードを上げています...'
        },
        {
            gif: '/static/images/rocket_orbit.gif', // ★★★ パスを直接記述 ★★★
            text: 'ようこそ宇宙へ！',
            subtext: '君だけの星座を探しています...'
        }
    ];

    let currentStage = 0;

    const loadingOverlay = document.getElementById('loading-overlay');
    const gifContainer = document.querySelector('.loading-gif-container');
    const mainText = document.querySelector('.loading-text');
    const subText = document.querySelector('.loading-sub-text');

    gifContainer.style.backgroundImage = `url(${loadingStages[currentStage].gif})`;
    mainText.textContent = loadingStages[currentStage].text;
    subText.textContent = loadingStages[currentStage].subtext;
    
    loadingOverlay.style.display = 'flex';

    const stageInterval = setInterval(() => {
        currentStage = (currentStage + 1) % loadingStages.length;
        gifContainer.style.backgroundImage = `url(${loadingStages[currentStage].gif})`;
        mainText.textContent = loadingStages[currentStage].text;
        subText.textContent = loadingStages[currentStage].subtext;
    }, 4000);

    setTimeout(function() {
        document.querySelector('form').submit();
    }, 1000);

    return false;
}

// 神話の紙芝居機能
document.addEventListener('DOMContentLoaded', () => {
    const mythStoryDiv = document.getElementById('myth-story');
    const navButtonsContainer = document.querySelector('.navigation-buttons');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');

    if (!mythStoryDiv) return;

    const fullMythText = mythStoryDiv.textContent.trim();
    const pages = fullMythText.split('。').map(s => s.trim()).filter(s => s.length > 0);

    let currentPage = 0;

    function renderPage() {
        if (pages.length > 0) {
            mythStoryDiv.textContent = pages[currentPage] + '。';
            nextButton.style.visibility = (currentPage === pages.length - 1) ? 'hidden' : 'visible';
            prevButton.style.visibility = (currentPage === 0) ? 'hidden' : 'visible';
        }
    }
    
    if (pages.length > 1) {
        navButtonsContainer.style.display = 'flex';
        renderPage();
    } else {
        navButtonsContainer.style.display = 'none';
    }
    
    prevButton.addEventListener('click', () => {
        if (currentPage > 0) { currentPage--; renderPage(); }
    });

    nextButton.addEventListener('click', () => {
        if (currentPage < pages.length - 1) { currentPage++; renderPage(); }
    });
});