<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "validate";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if (isset($_POST['submit'])) {
    $adminName = $_POST['admin-name'];
    $adminPassword = $_POST['admin-password'];

    $sql = "SELECT * FROM admin WHERE admin_name = '$adminName' AND admin_password = '$adminPassword'";
    $result = $conn->query($sql);

    if ($result->num_rows == 1) {
        session_start();
        $_SESSION['adminName'] = $adminName;
        header("Location: analysis.php");
        exit();
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
			<form method="POST" action="" class="login-form">
				<label for="chk" aria-hidden="true">Admin Login</label>
				<input type="text" name="admin-name" placeholder="Admin name" required>
				<div class="password-wrapper">
					<input type="password" name="admin-password" id="admin-password" placeholder="Password" required>
				</div>
				<button type="submit" name="submit" value="Login">login</button>
			</form>
		</div>

		<!-- <div class="login">
			<form method="POST" action="index.php" class="user-login-form">
				<label for="chk" aria-hidden="true">User Login</label>
				<button>Click here.!</button>
			</form>
		</div> -->
	</div>

	
</body>
</html>
