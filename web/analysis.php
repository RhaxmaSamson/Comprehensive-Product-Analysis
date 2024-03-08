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
    <h1>E-commerce analysis of electronic products</h1>
    <div style="width: 50%;">
        <canvas id="barChart"></canvas>
    </div>
    <br>
    <div style="width: 30%;">
        <canvas id="pieChart"></canvas>
    </div>

    <?php
        // Your PHP code to fetch data (replace this with actual data retrieval)
        $labels = ['USB-C Charging Cable', 'Lightning Charging Cable', 'Google Phone', 'iPhone', 'Wired Headphones', 'Apple Airpods Headphones', 'Bose SoundSport Headphones'];
        $values = [11.78, 11.65, 2.97, 3.68, 10.16, 8.36, 7.17];

        // Convert PHP arrays to JavaScript arrays
        $labels_js = json_encode($labels);
        $values_js = json_encode($values);
    ?>

    <script>
        // Bar Chart
        var ctxBar = document.getElementById('barChart').getContext('2d');
        var barChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: <?php echo $labels_js; ?>,
                datasets: [{
                    label: 'Bar Chart',
                    data: <?php echo $values_js; ?>,
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
                labels: <?php echo $labels_js; ?>,
                datasets: [{
                    data: <?php echo $values_js; ?>,
                    backgroundColor: ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow']
                }]
            }
        });
    </script>
</body>
</html>
