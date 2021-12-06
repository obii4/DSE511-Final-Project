from sklearn.model_selection import train_test_split

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.15, shuffle=True, stratify=y, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                 test_size=0.15/0.85, shuffle=True, stratify=y_train, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
