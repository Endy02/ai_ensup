from ai.core.cleaner import Cleaner
from ai.core.analyzer import Analizer

if __name__ == "__main__":
    # Clean data
    cleaner = Cleaner()
    cleaner.generate_final_dataset()

    # Analyze data
    analyzer = Analizer()
    analyzer.model_selection() # Model Selection
    analyzer.opitimization()
    
    # loader = Loader()
    # df_2015 = loader.get_sea_bench_2015()
    # df_2016 = loader.get_sea_bench_2016()
    # print(f"Final dataset shape : {df_final.shape}")
    # print(f"SEA Benchmarking 2015 dataset shape : {df_2015.shape}")
    # print(f"SEA Benchmarking 2016 dataset shape : {df_2016.shape}")