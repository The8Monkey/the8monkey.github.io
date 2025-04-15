"use strict";

let classifier;
let label = "";
let confidence = "";
let chart;
const dropzone = document.getElementById("dropzone");

const init = () => {
    preload();
    setup();
    dropzone.addEventListener("drop", function (evt) {
        let dt = evt.dataTransfer;
        let file = dt.files[0];
        if(file.type === 'image/png' || file.type === 'image/jpeg'){
            handleFiles(file);
        }
    });
};

const preload = () => {
    classifier = ml5.imageClassifier("MobileNet");
};

const setup = () => {
    const imageArray = document.querySelectorAll("img");
    imageArray.forEach((img) => {        
        classifier.classify(img).then((results) => {
            createChart(img.id, results)
        });
    });
};

const createChart = (id, results) => {
    let ctx = document.getElementById(id + "Chart").getContext("2d");
    return new Chart(ctx, {
        type: "bar",
        data: {
            labels: [
                results[0].label
                    .split(",")
                    .concat(Math.round(results[0].confidence * 100) + "%"),
                results[1].label
                    .split(",")
                    .concat(Math.round(results[1].confidence * 100) + "%"),
                results[2].label
                    .split(",")
                    .concat(Math.round(results[2].confidence * 100) + "%"),
            ],
            datasets: [
                {
                    label: "Confidence",
                    data: [
                        Math.round(results[0].confidence * 10000) / 10000,
                        Math.round(results[1].confidence * 10000) / 10000,
                        Math.round(results[2].confidence * 10000) / 10000,
                    ],
                    backgroundColor: "rgba(33, 95, 31, 0.5)",
                    borderColor: "rgb(61, 165, 84)",
                    borderWidth: 1,
                },
            ],
        },
        options: {
            scales: {
                yAxes: [
                    {
                        ticks: {
                            suggestedMin: 0,
                        },
                    },
                ],
            },
        },
    });
};

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(evt) {
    evt.preventDefault();
    evt.stopPropagation();
}

const handleFiles = (file) => {
    console.log(file);
    
    previewFile(file);
    document.getElementById("uploadRow").classList.remove("hidden");
};

const previewFile = (file) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function () {
        let img = document.createElement("img");
        img.src = reader.result;
        document.getElementById("preview").innerHTML = "";
        document.getElementById("preview").append(img);
        if(chart){
            chart.destroy();
        }
        classifier.classify(img).then((results) => {   
            chart = createChart("upload", results)
        });
    };
};

init();
