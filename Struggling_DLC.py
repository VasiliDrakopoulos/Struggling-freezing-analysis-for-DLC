import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class OptimizedStruggleIndex:
    
    def __init__(self, animals_per_video=1, sensitivity=0.6, fps=30):

        self.animals_per_video = animals_per_video
        self.sensitivity = max(0.1, min(1.0, sensitivity)) 
        self.fps = fps
        
        self.confidence_drop_threshold = 0.10 / max(0.1, sensitivity)
        self.large_change_threshold = 0.05 / max(0.1, sensitivity)
        
        self.min_bout_duration = 0.8
        
        self.feature_weights = {
            'instability': 0.40, 
            'drop_intensity': 0.35,
            'low_confidence': 0.25
        }
        
        self.intensity_scaling_factor = 0.3
        
        print(f"STRUGGLE INDEX")
        print(f"Sensitivity: {sensitivity}")
        print(f"Min bout duration: {self.min_bout_duration}s")
        print(f"Expected struggle: 20-60% of video")
        print(f"Expected intensities: 0.2-3.0")
        
    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        
        available_bodyparts = []
        for col in df.columns:
            if col.endswith('_confidence') and not col.startswith('frame'):
                bp_name = col.replace('_confidence', '')
                available_bodyparts.append(bp_name)
        
        print(f"Loaded {len(df)} frames, {len(available_bodyparts)} body parts")
        return df, available_bodyparts
    
    def calculate_intensity_features(self, df, available_bodyparts):
        features = pd.DataFrame(index=df.index)
        
        drop_intensities = []
        
        for bp in available_bodyparts:
            conf_col = f'{bp}_confidence'
            if conf_col in df.columns:
                conf_data = df[conf_col].fillna(method='ffill').fillna(method='bfill')
                
                drops = -conf_data.diff()
                drops = drops.clip(lower=0)
                
                amplified_drops = drops * (1 + drops * 3)
                drop_intensities.append(pd.Series(amplified_drops, name=f'{bp}_drops'))
        
        if drop_intensities:
            drop_df = pd.concat(drop_intensities, axis=1)
            features['drop_intensity'] = drop_df.sum(axis=1)
        
        instability_features = []
        
        for bp in available_bodyparts:
            conf_col = f'{bp}_confidence'
            if conf_col in df.columns:
                conf_data = df[conf_col].fillna(method='ffill').fillna(method='bfill')
                
                second_deriv = np.abs(conf_data.diff().diff())
                instability = second_deriv * 7
                instability_features.append(pd.Series(instability, name=f'{bp}_instability'))
        
        if instability_features:
            instab_df = pd.concat(instability_features, axis=1)
            features['instability'] = instab_df.sum(axis=1)
        
        low_conf_features = []
        
        for bp in available_bodyparts:
            conf_col = f'{bp}_confidence'
            if conf_col in df.columns:
                conf_data = df[conf_col].fillna(method='ffill')
                low_conf = (1 - conf_data)
                low_conf_penalty = low_conf * 2.0
                low_conf_features.append(pd.Series(low_conf_penalty, name=f'{bp}_lowconf'))
        
        if low_conf_features:
            low_conf_df = pd.concat(low_conf_features, axis=1)
            features['low_confidence'] = low_conf_df.sum(axis=1)
        
        features = features.fillna(0)
        
        for col in features.columns:
            features[col] = features[col] * self.intensity_scaling_factor
        
        return features
    
    def smooth_and_amplify(self, features):
        smoothed = features.copy()
        
        for col in features.columns:
            series = features[col]
            if len(series) > 7:
                try:
                    filled = series.fillna(method='ffill').fillna(method='bfill')
                    smoothed_series = savgol_filter(filled, 7, 3)
                    smoothed[col] = smoothed_series
                except:
                    smoothed[col] = series
            else:
                smoothed[col] = series
        
        return smoothed
    
    def calculate_struggle_intensity(self, features):
        intensity = pd.Series(0, index=features.index)
        total_weight = 0
        
        for feature, weight in self.feature_weights.items():
            if feature in features.columns:
                feat_vals = features[feature]
                if feat_vals.max() > 0:
                    scaled = (feat_vals / feat_vals.max()) * 2.0
                else:
                    scaled = feat_vals
                
                intensity += scaled * weight
                total_weight += weight
        
        if total_weight > 0:
            intensity = intensity / total_weight
        
        intensity = intensity * (1.0 + (1.0 - self.sensitivity) * 0.5)
        
        intensity = intensity * (1 + intensity * 0.3)
        
        return intensity
    
    def detect_struggle_periods(self, intensity):
        if intensity.max() == 0:
            return 0.05, []
        
        mean_intensity = intensity.mean()
        
        q60 = np.percentile(intensity, 60)
        threshold = q60 * 0.9
        
        threshold = threshold * (0.8 + (self.sensitivity - 0.5) * 0.4)
        
        threshold = max(threshold, mean_intensity * 1.1)
        threshold = min(threshold, mean_intensity * 2.5)
        threshold = max(threshold, 0.03)
        
        struggle_frames = intensity > threshold
        
        min_gap = int(self.fps * 0.3)
        
        changes = np.diff(np.concatenate(([0], struggle_frames.astype(int), [0])))
        bout_starts = np.where(changes == 1)[0]
        bout_ends = np.where(changes == -1)[0] - 1
        
        merged_starts = []
        merged_ends = []
        
        if len(bout_starts) > 0:
            current_start = bout_starts[0]
            current_end = bout_ends[0]
            
            for i in range(1, len(bout_starts)):
                if bout_starts[i] - current_end <= min_gap:
                    current_end = bout_ends[i]
                else:
                    merged_starts.append(current_start)
                    merged_ends.append(current_end)
                    current_start = bout_starts[i]
                    current_end = bout_ends[i]
            
            merged_starts.append(current_start)
            merged_ends.append(current_end)
        
        valid_bouts = []
        for start, end in zip(merged_starts, merged_ends):
            duration = (end - start + 1) / self.fps
            
            if duration >= self.min_bout_duration:
                bout_intensity = intensity.iloc[start:end+1]
                
                if bout_intensity.mean() > threshold * 0.6:
                    valid_bouts.append({
                        'start_frame': start,
                        'end_frame': end,
                        'duration_seconds': duration,
                        'mean_intensity': bout_intensity.mean(),
                        'max_intensity': bout_intensity.max(),
                        'start_time': start / self.fps,
                        'end_time': end / self.fps,
                        'total_intensity': bout_intensity.sum() * duration
                    })
        
        return threshold, valid_bouts
    
    def debug_intensity_distribution(self, intensity, threshold):
        print(f"DEBUG - Intensity distribution:")
        print(f"Mean: {intensity.mean():.4f}")
        print(f"Std: {intensity.std():.4f}")
        print(f"Min: {intensity.min():.4f}")
        print(f"Max: {intensity.max():.4f}")
        
        for p in [50, 60, 70, 75, 80, 90]:
            q = np.percentile(intensity, p)
            print(f"     {p}th percentile: {q:.4f}")
        
        print(f"Current threshold: {threshold:.4f}")
        above_threshold = (intensity > threshold).sum()
        percent_above = above_threshold / len(intensity) * 100
        print(f"Frames above threshold: {above_threshold}/{len(intensity)} ({percent_above:.1f}%)")
    
    def analyze_video(self, csv_file):
        print(f"\n{'='*60}")
        print(f"ANALYZING: {os.path.basename(csv_file)}")
        
        df, available_bodyparts = self.load_data(csv_file)
        
        if not available_bodyparts:
            print("No body parts found!")
            return None
        
        features = self.calculate_intensity_features(df, available_bodyparts)
        features = self.smooth_and_amplify(features)
        
        intensity = self.calculate_struggle_intensity(features)
        
        threshold, struggle_bouts = self.detect_struggle_periods(intensity)
        
        self.debug_intensity_distribution(intensity, threshold)
        
        total_video_time = len(df) / self.fps
        total_struggle_time = sum(b['duration_seconds'] for b in struggle_bouts)
        weighted_struggle = sum(b['total_intensity'] for b in struggle_bouts)
        
        fixed_weighted_struggle = sum(b['mean_intensity'] * b['duration_seconds'] for b in struggle_bouts)
        
        results = {
            'video': os.path.basename(csv_file),
            'animals': self.animals_per_video,
            'sensitivity': self.sensitivity,
            'total_frames': len(df),
            'video_seconds': total_video_time,
            'mean_intensity': intensity.mean(),
            'max_intensity': intensity.max(),
            'intensity_std': intensity.std(),
            'threshold': threshold,
            'struggle_bouts': len(struggle_bouts),
            'total_struggle_seconds': total_struggle_time,
            'weighted_struggle': weighted_struggle,
            'fixed_weighted_struggle': fixed_weighted_struggle,
            'struggle_percentage': (total_struggle_time / total_video_time * 100),
            'intensity_weighted_percentage': (weighted_struggle / total_video_time * 100),
            'fixed_weighted_percentage': (fixed_weighted_struggle / total_video_time * 100),
            'mean_bout_intensity': np.mean([b['mean_intensity'] for b in struggle_bouts]) if struggle_bouts else 0,
            'mean_bout_duration': np.mean([b['duration_seconds'] for b in struggle_bouts]) if struggle_bouts else 0,
            'max_bout_duration': np.max([b['duration_seconds'] for b in struggle_bouts]) if struggle_bouts else 0,
            'min_bout_duration': np.min([b['duration_seconds'] for b in struggle_bouts]) if struggle_bouts else 0,
            'bout_details': struggle_bouts,
            'raw_intensity': intensity.values
        }
        
        print(f"RESULTS:")
        print(f"Video: {results['video_seconds']:.0f}s ({results['video_seconds']/60:.1f} min)")
        print(f"Mean intensity: {results['mean_intensity']:.3f} (target: 0.2-0.8)")
        print(f"Max intensity: {results['max_intensity']:.3f} (target: 0.5-3.0)")
        print(f"Threshold: {results['threshold']:.3f}")
        print(f"Struggle bouts: {results['struggle_bouts']}")
        print(f"Struggle time: {results['total_struggle_seconds']:.0f}s ({results['struggle_percentage']:.1f}%)")
        print(f"Fixed weighted %: {results['fixed_weighted_percentage']:.1f}% (intensity × duration)")
        if results['struggle_bouts'] > 0:
            print(f"Avg bout: {results['mean_bout_duration']:.2f}s (min: {results['min_bout_duration']:.2f}s)")
        
        self.validate_results(results, total_video_time)
        
        return results
    
    def validate_results(self, results, total_video_time):
        print(f"   VALIDATION (Updated expectations):")
        
        if results['mean_intensity'] > 1.0:
            print(f"Mean intensity high: {results['mean_intensity']:.3f} (should be <1.0)")
        elif results['mean_intensity'] < 0.1:
            print(f"Mean intensity low: {results['mean_intensity']:.3f} (should be >0.1)")
        else:
            print(f"Mean intensity OK: {results['mean_intensity']:.3f}")
        
        if results['max_intensity'] > 5.0:
            print(f"Max intensity very high: {results['max_intensity']:.3f}")
        elif results['max_intensity'] < 0.3:
            print(f"Max intensity low: {results['max_intensity']:.3f}")
        else:
            print(f"Max intensity OK: {results['max_intensity']:.3f}")
        
        expected_min = total_video_time * 0.20
        expected_max = total_video_time * 0.70
        
        if results['total_struggle_seconds'] < expected_min:
            print(f"Low struggle: {results['struggle_percentage']:.1f}% (<20% expected)")
        elif results['total_struggle_seconds'] > expected_max:
            print(f"High struggle: {results['struggle_percentage']:.1f}% (>70% possible)")
        else:
            print(f"Struggle time OK: {results['struggle_percentage']:.1f}%")
        
        if results['mean_bout_duration'] < 0.8:
            print(f"Short bouts: {results['mean_bout_duration']:.2f}s (<0.8s min)")
        elif results['mean_bout_duration'] > 5.0:
            print(f"Long bouts: {results['mean_bout_duration']:.2f}s (>5.0s)")
        else:
            print(f"Bout duration OK: {results['mean_bout_duration']:.2f}s")
    
    def run_batch_analysis(self, folder_path, sensitivity=0.6):
        os.chdir(folder_path)
        csv_files = glob.glob("*_coordinates.csv")
        
        print(f"Found {len(csv_files)} CSV files")
        print(f"IMPROVED settings:")
        print(f"Sensitivity: {sensitivity}")
        print(f"Animals per video: {self.animals_per_video}")
        print(f"Intensity scaling: {self.intensity_scaling_factor}x")
        print(f"Min bout duration: {self.min_bout_duration}s")
        print(f"Expected struggle: 20-60%")
        
        self.sensitivity = sensitivity
        
        all_results = []
        
        for csv_file in csv_files:
            results = self.analyze_video(csv_file)
            
            if results:
                all_results.append(results)
                
                plot_file = csv_file.replace('.csv', f'_improved_s{sensitivity}_min{self.min_bout_duration}s.png')
                self.plot_results(results, save_path=plot_file)
        
        if all_results:
            summary = []
            for r in all_results:
                summary.append({
                    'video': r['video'],
                    'sensitivity': r['sensitivity'],
                    'video_seconds': r['video_seconds'],
                    'video_minutes': r['video_seconds'] / 60,
                    'mean_intensity': r['mean_intensity'],
                    'max_intensity': r['max_intensity'],
                    'intensity_std': r['intensity_std'],
                    'threshold': r['threshold'],
                    'struggle_bouts': r['struggle_bouts'],
                    'total_struggle_seconds': r['total_struggle_seconds'],
                    'struggle_percentage': r['struggle_percentage'],
                    'weighted_struggle_percentage': r['intensity_weighted_percentage'],
                    'fixed_weighted_percentage': r['fixed_weighted_percentage'],
                    'mean_bout_duration': r['mean_bout_duration'],
                    'mean_bout_intensity': r['mean_bout_intensity'],
                    'max_bout_duration': r['max_bout_duration'],
                    'min_bout_duration': r['min_bout_duration']
                })
            
            summary_df = pd.DataFrame(summary)
            
            print(f"\n{'='*60}")
            print("IMPROVED BATCH SUMMARY:")
            print(summary_df.to_string())
            
            print(f"\nFINAL VALIDATION:")
            avg_intensity = summary_df['mean_intensity'].mean()
            avg_max_intensity = summary_df['max_intensity'].mean()
            avg_struggle = summary_df['struggle_percentage'].mean()
            
            print(f"Avg mean intensity: {avg_intensity:.3f} {'✓' if 0.1 <= avg_intensity <= 1.0 else '⚠️'}")
            print(f"Avg max intensity: {avg_max_intensity:.3f} {'✓' if 0.3 <= avg_max_intensity <= 5.0 else '⚠️'}")
            print(f"Avg struggle: {avg_struggle:.1f}% {'✓' if 20 <= avg_struggle <= 60 else '⚠️'}")
            print(f"Total bouts: {summary_df['struggle_bouts'].sum()}")
            print(f"Avg bout duration: {summary_df['mean_bout_duration'].mean():.2f}s")
            print(f"Min bout duration used: {self.min_bout_duration}s")
            
            summary_file = os.path.join(folder_path, f'improved_summary_s{sensitivity}_min{self.min_bout_duration}s.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary saved: {summary_file}")
            
            return all_results, summary_df
        
        return [], pd.DataFrame()
    
    def plot_results(self, results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        time_seconds = np.arange(len(results['raw_intensity'])) / self.fps
        
        axes[0, 0].plot(time_seconds, results['raw_intensity'], 
                       linewidth=0.5, alpha=0.7, color='blue')
        axes[0, 0].axhline(y=results['threshold'], color='red', 
                          linestyle='--', alpha=0.7, linewidth=1.5,
                          label=f'Threshold: {results["threshold"]:.3f}')
        
        for bout in results['bout_details']:
            axes[0, 0].axvspan(bout['start_time'], bout['end_time'], 
                             alpha=0.2, color='red')
        
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Struggle Intensity')
        axes[0, 0].set_title(f'{results["video"]}\n'
                           f'Sensitivity: {results["sensitivity"]} | '
                           f'Struggle: {results["struggle_percentage"]:.1f}% | '
                           f'Min bout: {self.min_bout_duration}s')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        max_show = min(3.0, np.max(results['raw_intensity']) * 1.1)
        axes[0, 1].hist(results['raw_intensity'], bins=100, 
                       alpha=0.7, edgecolor='black', density=True,
                       range=(0, max_show))
        axes[0, 1].axvline(x=results['threshold'], color='red', 
                          linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Struggle Intensity (0-3)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'Intensity Distribution\n'
                           f'Mean: {results["mean_intensity"]:.3f} ± {results["intensity_std"]:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        if results['bout_details']:
            durations = [b['duration_seconds'] for b in results['bout_details']]
            intensities = [b['mean_intensity'] for b in results['bout_details']]
            
            scatter = axes[1, 0].scatter(durations, intensities, 
                                        c=intensities, cmap='viridis', 
                                        alpha=0.7, s=50, edgecolors='black')
            axes[1, 0].set_xlabel('Bout Duration (seconds)')
            axes[1, 0].set_ylabel('Mean Bout Intensity')
            axes[1, 0].set_title(f'{len(durations)} Struggle Bouts (≥{self.min_bout_duration}s)\n'
                               f'Avg: {results["mean_bout_duration"]:.2f}s, '
                               f'Intensity: {results["mean_bout_intensity"]:.3f}')
            axes[1, 0].axvline(x=self.min_bout_duration, color='gray', linestyle=':', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='Intensity')
        
        cumulative = np.cumsum(results['raw_intensity'] > results['threshold']) / self.fps
        axes[1, 1].plot(time_seconds, cumulative, linewidth=2, color='darkred')
        axes[1, 1].fill_between(time_seconds, 0, cumulative, alpha=0.3, color='red')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Cumulative Struggle (seconds)')
        axes[1, 1].set_title(f'Cumulative Struggle\n'
                           f'Total: {results["total_struggle_seconds"]:.0f}s '
                           f'({results["struggle_percentage"]:.1f}%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Plot saved: {save_path}")
        
        plt.show()

if __name__ == "__main__":
    FOLDER_PATH = r"D:\Deeplabcut\RestraintStress2\HFDNaChBac"
    
    print("=" * 60)
    print("STRUGGLING")
    print("=" * 60)
    print("Key improvements:")
    print("   1. Minimum bout duration: 0.8s")
    print("   2. Increased intensity scaling")
    print("   3. Lower detection tresholds (sensitivity)")
    print("   4. Preserves brief signals")
    print("   5. Weighted percentage")
    print("=" * 60)
    
    sensitivities_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    optimal_sensitivity = None
    
    for sensitivity in sensitivities_to_test:
        print(f"\n{'='*60}")
        print(f"Testing sensitivity: {sensitivity}")
        
        analyzer = OptimizedStruggleIndex(
            animals_per_video=1,
            sensitivity=sensitivity,
            fps=30
        )
        
        os.chdir(FOLDER_PATH)
        csv_files = glob.glob("*_coordinates.csv")[:2]
        
        test_results = []
        for csv_file in csv_files:
            results = analyzer.analyze_video(csv_file)
            if results:
                test_results.append(results)
        
        if test_results:
            avg_struggle = np.mean([r['struggle_percentage'] for r in test_results])
            avg_intensity = np.mean([r['mean_intensity'] for r in test_results])
            print(f"   Avg struggle: {avg_struggle:.1f}%, Avg intensity: {avg_intensity:.3f}")
            
            if 20 <= avg_struggle <= 60 and 0.1 <= avg_intensity <= 1.0:
                print(f"Approved sensitivity")
                optimal_sensitivity = sensitivity
                break
    
    if optimal_sensitivity is None:
        print(f"\nNo optimal sensitivity found in testing")
        print(f"Using middle sensitivity: 0.5")
        optimal_sensitivity = 0.5
    
    print(f"\n{'='*60}")
    print(f"Running final analysis with sensitivity: {optimal_sensitivity}")
    print(f"Minimum bout duration: 0.8s")
    print(f"{'='*60}")
    
    analyzer = OptimizedStruggleIndex(
        animals_per_video=1,
        sensitivity=optimal_sensitivity,
        fps=30
    )
    
    results, summary = analyzer.run_batch_analysis(
        folder_path=FOLDER_PATH,
        sensitivity=optimal_sensitivity
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)