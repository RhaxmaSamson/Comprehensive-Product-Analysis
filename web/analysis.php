<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis</title>
    <!-- Include Chart.js from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1> E-commerce analysis of electronic products </h1>
    <div style="width: 50%;">
        <canvas id="barChart"></canvas>
    </div>
    <br>
    <div style="width: 30%;">
        <canvas id="pieChart"></canvas>
    </div>

    <script>
        // Your PHP data (replace this with actual data)
        var labels = ['USB-C Charging Cable', 'Lightning Charging Cable', 'Google Phone', 'iPhone', 'Wired Headphones', 'Apple Airpods Headphones', 'Bose SoundSport Headphones'];
        var values = [11.78, 11.65, 2.97, 3.68, 10.16, 8.36, 7.17];

        // Bar Chart
        var ctxBar = document.getElementById('barChart').getContext('2d');
        var barChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Bar Chart',
                    data: values,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Pie Chart
        var ctxPie = document.getElementById('pieChart').getContext('2d');
        var pieChart = new Chart(ctxPie, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow']
                }]
            }
        });
    </script>
</body>
</html>
