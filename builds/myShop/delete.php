<?php

// Check if 'ID' is present in the URL
if (isset($_GET["id"]) ){

    $id = $_GET["id"];

    // Set up database connection
    $servername = "localhost";
    $username = "root";
    $password = "";
    $database = "myshop";

    // Connect to the database
    $connection = new mysqli($servername, $username, $password, $database);

    // Delete the client with matching last name
    $sql = "DELETE FROM characters WHERE id='$id'";
    $connection->query($sql);
}

// Redirect to main page after deletion
header("location: /programs/myShop/index.php");
exit;

?>
