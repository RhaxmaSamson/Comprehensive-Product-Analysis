<?php
session_start(); // Start session at the beginning of the PHP code

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "validate";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if (isset($_POST['submit'])) {
    // Prepare and bind parameters using prepared statements
    $sql = "SELECT * FROM admin WHERE admin_name = ? AND admin_password = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $adminName, $adminPassword);

    // Set parameters and execute
    $adminName = $_POST['admin-name'];
    $adminPassword = $_POST['admin-password'];
    $stmt->execute();

    // Get result
    $result = $stmt->get_result();

    if ($result->num_rows == 1) {
        // If login is successful, set session variable
        $_SESSION['adminName'] = $adminName;
        // Redirect to analysis page
        header("Location: analysis.php");
        exit(); // Ensure no further output is sent
    } else {
        $error = "Invalid admin name or password.";
    }
}

$conn->close();
?>



<!DOCTYPE html>
<html>
<head>
    <title>Login page</title>
    <link rel="stylesheet" type="text/css" href="login.css">
    <!-- <link href="https://fonts.googleapis.com/css2?family=Jost:wght@500&display=swap" rel="stylesheet"> -->
</head>
<body>
    <div class="main">
        <input type="checkbox" id="chk" aria-hidden="true">

        <div class="signup">
            <form method="POST" action="analysis.php" class="login-form">
                <label for="chk" aria-hidden="true">Admin Login</label>
                <input type="text" name="admin-name" placeholder="Admin name" required>
                <div class="password-wrapper">
                    <input type="password" name="admin-password" id="admin-password" placeholder="Password" required>
                </div>
                <button type="submit" name="submit" value="Login">login</button>
            </form>
        </div>

       
    </div>  
</body>
</html>
