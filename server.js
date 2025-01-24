const express = require("express");
const cors = require("cors");
const XLSX = require("xlsx");
const app = express();

// Activer CORS pour permettre les requêtes depuis le front-end
app.use(cors());

// Charger le fichier Excel une fois pour éviter de le relire à chaque requête
const workbook = XLSX.readFile("data.xlsx");

// Lire toutes les feuilles du fichier et les convertir en JSON
const sheetsData = {};
workbook.SheetNames.forEach(sheetName => {
    sheetsData[sheetName] = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);
});

// Endpoint pour récupérer les données par catégorie et discipline
app.get("/api/data", (req, res) => {
    const category = req.query.category || "all";
    const discipline = req.query.discipline || "all";

    // Filtrer les données par catégorie
    let data = category === "all" 
        ? Object.values(sheetsData).flat() 
        : sheetsData[category] || [];

    // Filtrer par discipline
    if (discipline !== "all") {
        data = data.filter(row => row["Discipline"] === discipline);
    }

    res.json(data);
});

// Démarrer le serveur
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Serveur démarré sur http://localhost:${PORT}`);
});
