"use strict";

let classifier;
// let img;
let label = "";
let confidence = "";

const init = () => {
    preload();
    setup();
};

const preload = () => {
    classifier = ml5.imageClassifier("MobileNet");
};

const setup = () => {
    const imageArray = document.querySelectorAll("img");
    console.log(imageArray);
    imageArray.forEach((img) => {
        classifier.classify(img).then((results) => {
            gotResult(results, img);
        });
    });
};

// Callback function for when classification has finished
const gotResult = (results, img) => {
    // The results are in an array ordered by confidence
    // console.log(results);
    // console.log(img);
    // console.log(img.id);

    // results.forEach((result) => {
    //     console.log(result);
    //     let elem = document.createElement("p");
    //     elem.innerHTML =
    //         'Ergebnis: <span class="big">' +
    //         result.label +
    //         '</span> mit confidence: <span class="big">' +
    //         result.confidence +
    //         "</span>";
    //     document.getElementById(img.id + "Result").appendChild(elem);
    // });
    createChart(img.id, results);
};

const createChart = (id, results) => {
    console.log(results);

    console.log( results[0].label.split(",").concat("test") )

    let ctx = document.getElementById(id + "Chart").getContext("2d");
    let chart = new Chart(ctx, {
        // The type of chart we want to create
        type: "bar",

        // The data for our dataset
        data: {
            labels: [
                results[0].label.split(",").concat(Math.round(results[0].confidence * 100) + "%"),
                results[1].label.split(",").concat(Math.round(results[1].confidence * 100) + "%"),
                results[2].label.split(",").concat(Math.round(results[2].confidence * 100) + "%"),
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

init();
