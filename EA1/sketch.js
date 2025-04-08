let classifier;
let img;
let label = "";
let confidence = "";

function preload() {
    classifier = ml5.imageClassifier("MobileNet");
    img = loadImage("images/hund.jpg");
}

function setup() {
    createCanvas(400, 400);
    classifier.classify(img, gotResult);
    image(img, 0, 0, width, height);
}

// Callback function for when classification has finished
function gotResult(results) {
    // The results are in an array ordered by confidence
    console.log(results);

    // Display the results on the canvas
    fill(255);
    stroke(0);
    textSize(18);
    label = "Label: " + results[0].label;
    confidence = "Confidence: " + nf(results[0].confidence, 0, 2);
    text(label, 10, 360);
    text(confidence, 10, 380);
}
