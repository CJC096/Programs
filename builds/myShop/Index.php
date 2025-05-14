<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Shop</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container myShop mt-5">

        <!-- Page heading -->
    <h2>List of Characters</h2>

    <!-- Button to go to create.php to add a new client -->
    <a class="btn btn-primary mb-3" href="/programs/myShop/create.php" role="button">Add Name</a>

    <!-- Table structure for displaying records from the database -->
    <table class="table table-bordered">
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>E-Mail</th>
                <th>Phone</th>
                <th>Address</th>
                <th>Created At</th>
                <th>Actions</th> <!-- Column for edit/delete actions -->
            </tr>
        </thead>
        <tbody>
        <?php
        // Set up database credentials
        $servername = "localhost";
        $username = "root";
        $password = "";
        $database = "myshop";

        // Create a new MySQL connection
        $connection = new mysqli($servername, $username, $password, $database);

        // If connection fails, stop the script and show an error
        if ($connection->connect_error) {
            die("Connection Failed: " . $connection->connect_error);
        }

        // SQL query to get all clients from the database
        $sql = "SELECT * FROM characters";
        $result = $connection->query($sql);

        // If query fails, stop script and show an error
        if (!$result) {
            die("Invalid Query: " . $connection->error);
        }

        // Loop through each row in the results and print it into the table
        while ($row = $result->fetch_assoc()) {
            echo "
            <tr>
                <td>$row[id]</td>
                <td>$row[name]</td>
                <td>$row[email]</td>
                <td>$row[phone]</td>
                <td>$row[address]</td>
                <td>$row[created_at]</td>
                <td>
                    <!-- Action buttons to edit or delete a client using the ID as a GET parameter -->
                    <a class='btn btn-primary btn-sm' href='/programs/myShop/update.php?id=$row[id]'>Edit</a>
                    <a class='btn btn-danger btn-sm' href='/programs/myShop/delete.php?id=$row[id]'>Delete</a>
                </td>
            </tr>
            ";
        }
        ?>
        </tbody>

    </div>
</body>
</html>