<?php
// Enable error reporting to assist with debugging
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Database connection configuration
$servername = "localhost";
$username = "root";
$password = "";
$database = "myshop";

// Create a new connection to the database
$connection = new mysqli($servername, $username, $password, $database);

// Initialize form input variables
$name = "";
$email = "";
$phone = "";
$address = "";

// Initialize messages for user feedback
$errorMessage = "";
$successMessage = "";

// Check if the form has been submitted via POST
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    // Retrieve data from the submitted form
    $name = $_POST['name'];
    $email = $_POST['email'];
    $phone = $_POST['phone'];
    $address = $_POST['address'];

    do {
        // Validate that all fields are filled in
        if (empty($name) || empty($email) || empty($phone) || empty($address)) {
            $errorMessage = "All fields are required";
            break;
        }

        // Escape special characters to help prevent SQL injection
        $name = $connection->real_escape_string($name);
        $email = $connection->real_escape_string($email);
        $phone = $connection->real_escape_string($phone);
        $address = $connection->real_escape_string($address);

        // SQL query to insert the new client record into the database
        $sql = "INSERT INTO characters (name, email, phone, address) 
                VALUES ('$name', '$email', '$phone', '$address')";

        $result = $connection->query($sql);

        // Check if the query was successful
        if (!$result) {
            $errorMessage = "Invalid Query: " . $connection->error;
            break;
        }

        // Set success message and redirect to the index page
        $successMessage = "Client added successfully";
        header("location: /programs/myShop/index.php");
        exit;

    } while (false); // Used to break out of the block on error
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

    <!-- Form to enter new client details -->
    <form method="post">
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
        <button type="submit" class="btn btn-primary">Add</button>
        <a href="/programs/myShop/index.php" class="btn btn-secondary">Cancel</a>
    </form>
</div>

</body>
</html>