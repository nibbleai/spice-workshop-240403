# New York City Taxi and Limousine Commission (TLC) trip demo project

## How to run the demo

**⚠️ You need Python >= 3.8**

0. Create a virtual env, and activate
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
    
2. Install dependencies
    ```bash
    # 'EXTRA_INDEX_URL' will be provided.
    python -m pip install -r requirements.txt --extra-index-url=${EXTRA_INDEX_URL}
    ```

3. Initialize spice store (Needs to be done only once)
    ```
    spice init store
    ```

4. Run the demo
    ```bash
    python -m src.main
    ```

5. Check the UI
    ```bash
    spice ui
    ```
