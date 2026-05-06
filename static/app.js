function classifyURL() {
    let url = document.getElementById("urlInput").value;
    let resultBox = document.getElementById("result");

    if (!url) {
        resultBox.innerHTML = "⚠️ Please enter a URL";
        return;
    }

    resultBox.innerHTML = "⏳ Checking...";

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: url })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultBox.innerHTML = "❌ Error: " + data.error;
        } else {
            resultBox.innerHTML = `Result: <b>${data.prediction}</b>`;
        }
    })
    .catch(err => {
        resultBox.innerHTML = "❌ Server error";
        console.log(err);
    });
}