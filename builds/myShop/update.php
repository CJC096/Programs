<?php
// Enable error reporting for debugging
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Database configuration
$servername = "localhost";
$username = "root";
$password = "";
$database = "myshop";

// Establish a new database connection
$connection = new mysqli($servername, $username, $password, $database);

// Initialize variables for form fields and feedback messages
$id = "";
$name = "";
$email = "";
$phone = "";
$address = "";

$errorMessage = "";
$successMessage = "";

// Check request method to determine action
if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    // If no ID is provided in the URL, redirect to main page
    if (!isset($_GET["id"])) {
        header("location: /programs/myShop/index.php");
        exit;
    }

    // Sanitize input from URL
    $id = $connection->real_escape_string($_GET["id"]);

    // Fetch existing character data using ID
    $sql = "SELECT * FROM characters WHERE id=$id";
    $result = $connection->query($sql);
    $row = $result->fetch_assoc();

    // If no matching record is found, redirect to main page
    if (!$row) {
        header("location: /programs/myShop/index.php");
        exit;
    }

    // Populate form fields with existing data
    $name = $row['name'];
    $email = $row['email'];
    $phone = $row['phone'];
    $address = $row['address'];

} else {
    // Handle form submission (POST request) for updating client

    // Retrieve and assign form data
    $id = $_POST['id'];
    $name = $_POST['name'];
    $email = $_POST['email'];
    $phone = $_POST['phone'];
    $address = $_POST['address'];

    do {
        // Validate that all fields are filled
        if (empty($id) || empty($name) || empty($email) || empty($phone) || empty($address)) {
            $errorMessage = "All fields are required";
            break;
        }

        // Sanitize form data before using in SQL query
        $name = $connection->real_escape_string($name);
        $email = $connection->real_escape_string($email);
        $phone = $connection->real_escape_string($phone);
        $address = $connection->real_escape_string($address);

        // Update query to modify the client's information
        $sql = "UPDATE characters SET 
                    name = '$name', 
                    email = '$email', 
                    phone = '$phone',
                    address = '$address'
                WHERE id = '$id'";

        $result = $connection->query($sql);

        // Handle query failure
        if (!$result) {
            $errorMessage = "Invalid Query: " . $connection->error;
            break;
        }

        // Success message
        $successMessage = "Character updated successfully";

        header("location: /programs/myShop/index.php");
        exit;

    } while (false);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
    
<div class="container mt-5">
    <h2>New Character</h2>

    <!-- Display an error message if one exists -->
    <?php if (!empty($errorMessage)) : ?>
        <div class="alert alert-danger"><?php echo $errorMessage; ?></div>
    <?php endif; ?>

    <!-- Form to update client details -->
    <form method="post">
        <input type="hidden" name = "id" value="<?php echo $id; ?>">
        <!-- Input for Name -->
        <div class="mb-3">
            <label class="form-label">Name</label>
            <input type="text" class="form-control" name="name" value="<?php echo $name; ?>">
        </div>

        <!-- Input for Email -->
        <div class="mb-3">
            <label class="form-label">E-Mail</label>
            <input type="email" class="form-control" name="email" value="<?php echo $email; ?>">
        </div>

        <!-- Input for Phone Number -->
        <div class="mb-3">
            <label class="form-label">Phone Number</label>
            <input type="text" class="form-control" name="phone" value="<?php echo $phone; ?>">
        </div>

        <!-- Input for Address -->
        <div class="mb-3">
            <label class="form-label">Address</label>
            <input type="text" class="form-control" name="address" value="<?php echo $address; ?>">
        </div>

        <!-- Submit and cancel buttons -->
        <button type="submit" class="btn btn-primary">Update</button>
        <a href="/programs/myShop/index.php" class="btn btn-secondary">Cancel</a>
    </form>
</div>

</body>
</html>