let classifier;
// let img;
let label = "";
let confidence = "";

function preload() {
    classifier = ml5.imageClassifier("MobileNet");
    // img = loadImage("images/hund.jpg");
}

function setup() {
    const imageArray = document.querySelectorAll("img");
    console.log(imageArray);
    imageArray.forEach(img => {
        classifier.classify(img).then(
            (results) => {gotResult(results, img)}
        )
    });
    // createCanvas(400, 400);
    // classifier.classify(img, gotResult);
    // image(img, 0, 0, width, height);
}

// Callback function for when classification has finished
function gotResult(results, img) {
    // The results are in an array ordered by confidence
    console.log(results);
    console.log(img);

    console.log(img.id);
    results.forEach(result =>{
        console.log(result);
        let elem = document.createElement("p")
        elem.innerHTML = "Ergebnis: " + result.label + " mit confidence: " + result.confidence; 
        document.getElementById(img.id+"Result").appendChild(elem);
    })
}
