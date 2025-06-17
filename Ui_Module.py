from load_dataset_module import DatasetLoader
from eda_module import DataExplorer, StrokeModelTrainer, ColumnEditor

class StrokeAnalysisCLI:

    def __init__(self, filepath):
        self.dataset_loader = DatasetLoader(filepath)
        self.df = self.dataset_loader.read_data()
        if self.df is None:
            print(" Unable to continue without data.")
            exit()
        self.explorer = DataExplorer(self.df)
        self.trainer = StrokeModelTrainer(self.df)

    def run(self):
        while True:
            print("\n-- Stroke Analysis CLI Menu ---")
            print("1. View Statistical Summary")
            print("2. Visualize Data")
            print("3. Handle Missing Values")
            print("4. View Class Distribution")
            print("5. Balance Data")
            print("6. Train ALL ML Models")
            print("7. Column Manipulation")
            print("8. Detect Outliers")
            print("9. Evaluate Each Model with Confusion Matrix")
            print("10.risk score")
            print("11 Save Dataset")
            print("12. Exit")

            choice = input("Enter your choice (1–12): ").strip()

            if choice == "1":
                print(self.explorer.statisticalSummary())

            elif choice == "2":
                self.explorer.visualize()

            elif choice == "3":
                self.df = self.explorer.handleMissingValues()
                missing = self.df.isnull().sum()
                remaining = missing[missing > 0]
                if remaining.empty:
                    print(" All missing values have been handled.")
                else:
                    print("Remaining missing values:")
                    print(remaining)

            elif choice == "4":
                target = input("Enter target column: ").strip()
                if target:
                    self.explorer.classDistribution(target)

            elif choice == "5":
                target = input("Enter target column to balance (e.g., stroke, chronic_stress, Income_level): ").strip()
                if target:
                    self.df = self.explorer.balanceData(target)
                    print(f" Data balanced on '{target}'.")

            elif choice == "6":
                self.trainer.trainAllTargets()
                print(" Model training complete.")

            elif choice == "7":
                self.columnManipulation()
            
            elif choice == "8":
                col = input("Enter column name to check for outliers: ")
                self.explorer.detect_outliers(col)

            elif choice == "9":
                target = input("Enter target column  as Chronic Stress,Physical Activity,Income Level,Stroke Occurrence")
                self.trainer.evaluateEachTarget(target)
                print(" Model training complete.")

            elif choice == "10":
                 self.explorer.compute_risk_score()
            
            elif choice == "11":
                self.explorer.saveToFile()
                print(" Dataset saved successfully.")

            elif choice == "12":
                print(" Exiting the application. Goodbye!")
                break

            else:
                print(" Invalid input. Please try again.")

    def columnManipulation(self):
        editor = ColumnEditor(self.df)
        while True:
            print("\n-- Column Manipulation ---")
            print("1. Add Column")
            print("2. Remove Column")
            print("3. Rename Column")
            print("4. Back to Main Menu")

            subchoice = input("Enter your choice (1–4): ").strip()

            if subchoice == "1":
                colname = input("Enter new column name: ").strip()
                if colname:
                    print(editor.addColumn(colname))
                    self.df = editor.getDataframe()

            elif subchoice == "2":
                colname = input("Enter column name to remove: ").strip()
                if colname:
                    print(editor.removeColumn(colname))
                    self.df = editor.getDataframe()

            elif subchoice == "3":
                old = input("Enter current column name: ").strip()
                new = input(f"Enter new name for '{old}': ").strip()
                if old and new:
                    print(editor.renameColumn(old, new))
                    self.df = editor.getDataframe()

            elif subchoice == "4":
                break

            else:
                print(" Invalid input. Please try again.")


# Main Execution
if __name__ == "__main__":
    csv_file_path = "stroke_dataset.csv"
    app = StrokeAnalysisCLI(csv_file_path)
    app.run()
