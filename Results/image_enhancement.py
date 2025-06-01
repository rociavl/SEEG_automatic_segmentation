import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt


class SEEGResultsAnalyzer:
    def __init__(self, output_dir=None):
        self.results_data = []
        self.summary_stats = {}
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.getcwd()  # Current working directory
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)  # Create if doesn't exist
        
        print(f"Output will be saved to: {self.output_dir}")
        
    def load_patient_data(self, patient_folders):
        """
        Load data for multiple patients
        patient_folders: dict like {'P1': {'fix': 'path/to/fix', 'ml': 'path/to/ml'}}
        """
        all_data = []
        
        for patient_id, folders in patient_folders.items():
            print(f"Processing {patient_id}...")
            
            for method_name, folder_path in folders.items():
                # Find all CSV files in the folder
                csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
                
                if not csv_files:
                    print(f"  Warning: No CSV files found in {folder_path}")
                    continue
                
                for csv_file in csv_files:
                    try:
                        # Load the CSV
                        df = pd.read_csv(csv_file)
                        
                        # Add metadata columns
                        df['Patient_ID'] = patient_id
                        df['Method'] = method_name
                        df['Source_File'] = os.path.basename(csv_file)
                        
                        all_data.append(df)
                        print(f"  Loaded {len(df)} records from {os.path.basename(csv_file)} ({method_name})")
                        
                    except Exception as e:
                        print(f"  Error loading {csv_file}: {e}")
        
        if all_data:
            self.combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal records loaded: {len(self.combined_data)}")
            return self.combined_data
        else:
            print("No data loaded!")
            return None
    
    def calculate_summary_stats(self):
        """Calculate summary statistics for each method"""
        if not hasattr(self, 'combined_data'):
            print("No data loaded. Run load_patient_data first.")
            return
        
        # Overall statistics by method
        method_stats = self.combined_data.groupby('Method').agg({
            'Success': ['count', 'sum', 'mean'],
            'Distance (mm)': ['mean', 'std', 'min', 'max'],
            'MSE': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns]
        
        # Add percentage within 2mm clinical threshold
        within_2mm = self.combined_data.groupby('Method').apply(
            lambda x: (x['Distance (mm)'] <= 2.0).mean() * 100
        ).round(1)
        method_stats['Within_2mm_percent'] = within_2mm
        
        self.method_summary = method_stats
        print("Method Summary Statistics:")
        print(self.method_summary)
        
        # Patient-specific statistics
        patient_method_stats = self.combined_data.groupby(['Patient_ID', 'Method']).agg({
            'Success': ['count', 'sum', 'mean'],
            'Distance (mm)': ['mean', 'std']
        }).round(3)
        
        self.patient_method_summary = patient_method_stats
        
        return method_stats
    
    def create_comparison_table(self):
        """Create a clean comparison table for thesis results"""
        if not hasattr(self, 'method_summary'):
            self.calculate_summary_stats()
        
        # Create clean table for thesis
        comparison_table = pd.DataFrame({
            'Total_Centroids': self.method_summary['Success_count'],
            'Success_Count': self.method_summary['Success_sum'],
            'Success_Rate_%': (self.method_summary['Success_mean'] * 100).round(1),
            'Mean_Distance_mm': self.method_summary['Distance (mm)_mean'].round(2),
            'Std_Distance_mm': self.method_summary['Distance (mm)_std'].round(2),
            'Within_2mm_%': self.method_summary['Within_2mm_percent'],
            'Mean_MSE': self.method_summary['MSE_mean'].round(4)
        })
        
        print("\n" + "="*60)
        print("THESIS RESULTS TABLE")
        print("="*60)
        print(comparison_table)
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, 'thesis_method_comparison.csv')
        comparison_table.to_csv(output_file)
        print(f"\nSaved to: {output_file}")
        
        return comparison_table
    
    def create_patient_breakdown(self):
        """Create patient-by-patient breakdown"""
        if not hasattr(self, 'combined_data'):
            return
        
        # Success rate by patient and method
        patient_success = self.combined_data.groupby(['Patient_ID', 'Method'])['Success'].mean().unstack(fill_value=0) * 100
        
        print("\n" + "="*60)
        print("SUCCESS RATE BY PATIENT AND METHOD (%)")
        print("="*60)
        print(patient_success.round(1))
        
        # Distance by patient and method
        patient_distance = self.combined_data.groupby(['Patient_ID', 'Method'])['Distance (mm)'].mean().unstack(fill_value=np.nan)
        
        print("\n" + "="*60)
        print("MEAN DISTANCE BY PATIENT AND METHOD (mm)")
        print("="*60)
        print(patient_distance.round(2))
        
        # Save both tables
        success_file = os.path.join(self.output_dir, 'patient_success_breakdown.csv')
        distance_file = os.path.join(self.output_dir, 'patient_distance_breakdown.csv')
        
        patient_success.to_csv(success_file)
        patient_distance.to_csv(distance_file)
        
        print(f"\nSaved breakdown tables to:")
        print(f"  {success_file}")
        print(f"  {distance_file}")
        
        return patient_success, patient_distance
    
    def create_visualizations(self):
        """Create visualizations for thesis"""
        if not hasattr(self, 'combined_data'):
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Success Rate by Method
        method_success = self.combined_data.groupby('Method')['Success'].mean() * 100
        axes[0,0].bar(method_success.index, method_success.values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0,0].set_title('Success Rate by Threshold Method')
        axes[0,0].set_ylabel('Success Rate (%)')
        axes[0,0].set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate(method_success.values):
            axes[0,0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. Distance Distribution by Method
        self.combined_data.boxplot(column='Distance (mm)', by='Method', ax=axes[0,1])
        axes[0,1].set_title('Distance Distribution by Method')
        axes[0,1].set_ylabel('Distance (mm)')
        axes[0,1].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2mm Clinical Threshold')
        axes[0,1].legend()
        
        # 3. Success Rate by Patient
        patient_success = self.combined_data.groupby(['Patient_ID', 'Method'])['Success'].mean().unstack()
        patient_success.plot(kind='bar', ax=axes[1,0], width=0.8)
        axes[1,0].set_title('Success Rate by Patient and Method')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_xlabel('Patient ID')
        axes[1,0].legend(title='Method')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Within 2mm Threshold by Method
        within_2mm = self.combined_data.groupby('Method').apply(
            lambda x: (x['Distance (mm)'] <= 2.0).mean() * 100
        )
        axes[1,1].bar(within_2mm.index, within_2mm.values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1,1].set_title('Clinical Success Rate (≤2mm) by Method')
        axes[1,1].set_ylabel('Percentage Within 2mm (%)')
        axes[1,1].set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(within_2mm.values):
            axes[1,1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'thesis_method_comparison_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: {plot_file}")
    
    def generate_thesis_text(self):
        """Generate formatted text for thesis results section"""
        if not hasattr(self, 'method_summary'):
            self.calculate_summary_stats()
        
        comparison_table = self.create_comparison_table()
        
        print("\n" + "="*80)
        print("FORMATTED TEXT FOR THESIS RESULTS SECTION")
        print("="*80)
        
        # Extract key numbers
        methods = comparison_table.index.tolist()
        best_method = comparison_table['Success_Rate_%'].idxmax()
        best_success_rate = comparison_table.loc[best_method, 'Success_Rate_%']
        best_distance = comparison_table.loc[best_method, 'Mean_Distance_mm']
        best_within_2mm = comparison_table.loc[best_method, 'Within_2mm_%']
        
        total_centroids = comparison_table['Total_Centroids'].iloc[0]  # Should be same for all methods
        
        print(f"""
7.3 Image Enhancement and Threshold Prediction Results

7.3.1 Threshold Method Comparison
Three threshold approaches were evaluated across all {len(self.combined_data['Patient_ID'].unique())} patients 
(total {total_centroids} centroids). {best_method} achieved the highest success rate 
({best_success_rate}%), followed by {comparison_table['Success_Rate_%'].nlargest(2).index[1]} 
({comparison_table['Success_Rate_%'].nlargest(2).iloc[1]}%) and {comparison_table['Success_Rate_%'].nsmallest(1).index[0]} 
({comparison_table['Success_Rate_%'].nsmallest(1).iloc[0]}%). Mean localization distances were 
{comparison_table.loc[best_method, 'Mean_Distance_mm']}±{comparison_table.loc[best_method, 'Std_Distance_mm']}mm, 
{comparison_table.loc[comparison_table['Success_Rate_%'].nlargest(2).index[1], 'Mean_Distance_mm']}±{comparison_table.loc[comparison_table['Success_Rate_%'].nlargest(2).index[1], 'Std_Distance_mm']}mm, 
and {comparison_table.loc[comparison_table['Success_Rate_%'].nsmallest(1).index[0], 'Mean_Distance_mm']}±{comparison_table.loc[comparison_table['Success_Rate_%'].nsmallest(1).index[0], 'Std_Distance_mm']}mm 
respectively (Table 7.X).

7.3.2 Clinical Accuracy Assessment
Within the 2mm clinical threshold, {best_method} correctly localized 
{best_within_2mm}% of electrodes, compared to {comparison_table.loc[comparison_table['Within_2mm_%'].nlargest(2).index[1], 'Within_2mm_%']}% 
for {comparison_table['Within_2mm_%'].nlargest(2).index[1]} and {comparison_table.loc[comparison_table['Within_2mm_%'].nsmallest(1).index[0], 'Within_2mm_%']}% 
for {comparison_table['Within_2mm_%'].nsmallest(1).index[0]} across the complete patient cohort.

7.3.3 Processing Efficiency
The threshold prediction pipeline demonstrated reliable performance across all 
patients, with {best_method} achieving superior accuracy in {len([p for p in self.combined_data['Patient_ID'].unique() if self.combined_data[(self.combined_data['Patient_ID']==p) & (self.combined_data['Method']==best_method)]['Success'].mean() == self.combined_data[self.combined_data['Patient_ID']==p].groupby('Method')['Success'].mean().max()])}/{len(self.combined_data['Patient_ID'].unique())} cases.
""")


# Usage example and main execution
def main():
    # Initialize analyzer with custom output directory
    output_directory = r"C:\Users\rocia\Downloads\TFG\Cohort\RESULTS_REPORT\IMAGE_GT"  
    analyzer = SEEGResultsAnalyzer(output_dir=output_directory)
    
    
    # Define patient folders - MODIFY THESE PATHS FOR YOUR DATA
    patient_folders = {
        'P1': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P1\P1_results\P1_fix_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P1\P1_results\P1_ml_greedy"
        },
        'P2': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P2\P2_results\P2_fix_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P2\P2_results\P2_ml_greedy"
        },

        'P3': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P3\P3_results\P3_fix_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P3\P3_results\P3_ml_greedy"
        },

        'P4': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P4\P4_results\P4_fix_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P4\P4_results\P4_ml_greedy"
        },

        'P5': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P5\P5_results\P5_fix_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P5\P5_results\P5_ml_greedy"
        },

        'P6': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P6\P6_results\fix_g",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P6\P6_results\ml_g"
        },

        'P7': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P7\P7_results\f_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P7\P7_results\ml_greedy"
        },

        'P8': {
            'fix': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P8\P8_RESULTS\f_greedy",
            'ml': r"C:\Users\rocia\Downloads\TFG\Cohort\Models\Centroids_model\DATA\P8\P8_RESULTS\m_g"
        },
    }
    
    
    print("SEEG Results Analysis Tool")
    print("=" * 50)
    
    # Load data
    data = analyzer.load_patient_data(patient_folders)
    
    if data is not None:
        # Calculate statistics
        analyzer.calculate_summary_stats()
        
        # Create comparison tables
        comparison_table = analyzer.create_comparison_table()
        
        # Create patient breakdown
        analyzer.create_patient_breakdown()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Generate thesis text
        analyzer.generate_thesis_text()
        
        print("\nAnalysis complete! Check the generated CSV files and plots.")
        
        return analyzer
    else:
        print("No data could be loaded. Please check your file paths.")
        return None

if __name__ == "__main__":
    analyzer = main()