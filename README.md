# New York City Taxi and Limousine Commission (TLC) trip demo project

## How to run the demo


1. Install dependencies
    ```bash
    # 'EXTRA_INDEX_URL' will be provided.
    pip install -r requirements.txt --extra-index-url=${EXTRA_INDEX_URL}
    ```

2. Initialize spice store (Needs to be done only once)
    ```
    spice init store
    ```

3. Run the demo
    ```bash
    python -m src.main
    ```

4. Check the UI
    ```bash
    spice ui
    ```
